"""
Joint Scaling Policy: Anchor / Burst server management with
intra-anchor replication, kickout migration, burst replication,
de-replication, and burst OFF logic.

Fully executable reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ------------------------------------------------------------------ #
#  Core data structures                                                #
# ------------------------------------------------------------------ #

@dataclass
class Session:
    """One active stream."""
    session_id: int
    video: int        # video index
    server: int       # server index currently serving this session
    bitrate: float    # r(v) snapshot at admission time


@dataclass
class Server:
    index: int
    capacity: float
    is_anchor: bool
    is_on: bool = True                         # anchors always True
    store: Set[int] = field(default_factory=set)  # videos stored


class SystemState:
    """Mutable global state shared by every function."""

    def __init__(
        self,
        anchors: List[Server],
        bursts: List[Server],
        bitrates: Dict[int, float],            # video -> per-stream bitrate
        margin: float = 0.0,
        off_threshold: float = 0.15,
    ):
        self.anchors: List[Server] = anchors
        self.bursts: List[Server] = bursts
        self.servers: Dict[int, Server] = {}
        for s in anchors + bursts:
            self.servers[s.index] = s

        self.bitrates = bitrates               # r(v)
        self.margin = margin
        self.off_threshold = off_threshold

        # RA[v], RB[v]  – rebuilt from Store
        self.RA: Dict[int, Set[int]] = {}      # video -> set of anchor indices
        self.RB: Dict[int, Set[int]] = {}      # video -> set of burst  indices
        self._rebuild_replica_maps()

        # active sessions
        self.sessions: List[Session] = []
        self._next_sid: int = 0

        # thresholds – caller must set per-video functions
        self.thA: Dict[int, float] = {}        # video -> anchor threshold
        self.thT: Dict[int, float] = {}        # video -> total  threshold

    # ---------- helpers ------------------------------------------------ #

    def _rebuild_replica_maps(self):
        self.RA.clear()
        self.RB.clear()
        for s in self.anchors:
            for v in s.store:
                self.RA.setdefault(v, set()).add(s.index)
        for s in self.bursts:
            for v in s.store:
                self.RB.setdefault(v, set()).add(s.index)

    def r(self, v: int) -> float:
        return self.bitrates.get(v, 0.0)

    def N(self, v: int, x: int) -> int:
        """Number of active streams of video *v* on server *x*."""
        return sum(1 for s in self.sessions if s.video == v and s.server == x)

    def U(self, x: int) -> float:
        """Current load on server *x*."""
        return sum(s.bitrate for s in self.sessions if s.server == x)

    def H(self, x: int) -> float:
        """Headroom of server *x*."""
        return self.servers[x].capacity - self.U(x)

    def B_ON(self) -> List[Server]:
        return [b for b in self.bursts if b.is_on]

    def idx(self, x: int) -> int:
        return x

    # ---------- primitives -------------------------------------------- #

    def policy_allows_store(self, server_idx: int, video: int) -> bool:
        """Override for custom storage policy."""
        return True

    def turn_on(self, b_idx: int):
        self.servers[b_idx].is_on = True

    def turn_off(self, b_idx: int):
        self.servers[b_idx].is_on = False

    def transmit_replica(self, v: int, src: int, tgt: int):
        self.servers[tgt].store.add(v)

    def delete_replica(self, v: int, x: int):
        self.servers[x].store.discard(v)

    def pick_one_active_session(self, v: int, src: int) -> Optional[Session]:
        for s in self.sessions:
            if s.video == v and s.server == src:
                return s
        return None

    def migrate_session(self, sess: Session, src: int, tgt: int):
        sess.server = tgt

    def serve_request(self, v: int, server: int, start_time: float) -> Session:
        s = Session(
            session_id=self._next_sid,
            video=v,
            server=server,
            bitrate=self.r(v),
        )
        self._next_sid += 1
        self.sessions.append(s)
        return s


# ------------------------------------------------------------------ #
#  Algorithm functions                                                 #
# ------------------------------------------------------------------ #

ACCEPT = True
REJECT = False


def joint_policy(state: SystemState, j: int, start_time: float) -> bool:
    """Main entry point. *j* is the video index of the incoming request."""
    v_j = j

    def _acA() -> float:
        return sum(state.H(a) for a in state.RA.get(v_j, set()))

    def _acT() -> float:
        acA = _acA()
        acB = sum(
            state.H(b)
            for b in state.RB.get(v_j, set())
            if state.servers[b].is_on
        )
        return acA + acB

    m = state.margin
    thA = state.thA.get(v_j, 0.0)
    thT = state.thT.get(v_j, 0.0)

    # ---- scaling decisions ----
    if _acA() <= thA - m:
        intra_anchor_replication(state, v_j)

        if _acA() <= thA - m:
            ok = kickout_one_session(state, v_j)

            if (not ok) or (_acA() <= thA - m):
                burst_replication(state, v_j)

    elif _acT() >= thT + m:
        de_replication(state, v_j, thT, m)

    # ---- Burst OFF rule ----
    b_highest = get_highest_index_on_burst(state)
    if b_highest is not None:
        rho = state.U(b_highest) / state.servers[b_highest].capacity if state.servers[b_highest].capacity > 0 else 0.0
        if rho < state.off_threshold:
            turn_off_burst_if_o1_feasible(state, b_highest)

    # ---- Admit & steer ----
    ra = state.RA.get(v_j, set())
    for a in ra:
        if state.H(a) >= state.r(v_j):
            a_star = select_anchor_for_request(state, v_j)
            if a_star is not None:
                state.serve_request(v_j, a_star, start_time)
                return ACCEPT
            break

    rb = state.RB.get(v_j, set())
    for b in sorted(rb):
        if state.servers[b].is_on and state.H(b) >= state.r(v_j):
            b_star = select_burst_for_request(state, v_j)
            if b_star is not None:
                state.serve_request(v_j, b_star, start_time)
                return ACCEPT
            break

    return REJECT


# ============================================================
# Stage A-2: Intra-Anchor Replication
# ============================================================

def intra_anchor_replication(state: SystemState, v_j: int):
    ra = state.RA.get(v_j, set())

    a_tgt: Optional[int] = None
    best_load = math.inf

    for srv in state.anchors:
        a = srv.index
        if a in ra:
            continue
        if not state.policy_allows_store(a, v_j):
            continue
        if state.H(a) < state.r(v_j):
            continue
        if state.U(a) < best_load:
            best_load = state.U(a)
            a_tgt = a

    if a_tgt is None:
        return

    # source: least-load anchor that already stores v_j
    a_src: Optional[int] = None
    min_load = math.inf
    for a in ra:
        if state.U(a) < min_load:
            min_load = state.U(a)
            a_src = a

    if a_src is None:
        return

    state.transmit_replica(v_j, a_src, a_tgt)
    state.servers[a_tgt].store.add(v_j)
    state.RA.setdefault(v_j, set()).add(a_tgt)


# ============================================================
# Stage B: Kickout exactly ONE donor stream
# ============================================================

def kickout_one_session(state: SystemState, v_j: int) -> bool:
    ra = state.RA.get(v_j, set())
    r_vj = state.r(v_j)

    best_found = False
    best_a: Optional[int] = None
    best_w: Optional[int] = None
    best_b: Optional[int] = None
    best_metric = -math.inf

    for a in ra:
        if state.H(a) >= r_vj:
            continue
        delta = r_vj - state.H(a)

        for w in list(state.servers[a].store):
            if w == v_j:
                continue
            n_wa = state.N(w, a)
            if n_wa <= 0:
                continue
            if state.r(w) < delta:
                continue

            rb_w = state.RB.get(w, set())
            b_on = state.B_ON()
            b_on_indices = {b.index for b in b_on}
            bcand = [
                b for b in rb_w
                if b in b_on_indices and state.H(b) >= state.r(w)
            ]
            if not bcand:
                continue

            metric = n_wa
            b0 = min(bcand)

            update = False
            if metric > best_metric:
                update = True
            elif metric == best_metric:
                if best_b is None or state.idx(b0) < state.idx(best_b):
                    update = True
                elif best_b is not None and state.idx(b0) == state.idx(best_b):
                    if best_w is None or state.r(w) > state.r(best_w):
                        update = True
                    elif best_w is not None and state.r(w) == state.r(best_w):
                        if best_a is None or state.idx(a) < state.idx(best_a):
                            update = True

            if update:
                best_metric = metric
                best_a = a
                best_w = w
                best_b = b0
                best_found = True

    if not best_found or best_a is None or best_w is None or best_b is None:
        return False

    s = state.pick_one_active_session(best_w, best_a)
    if s is None:
        return False

    state.migrate_session(s, best_a, best_b)
    return True


# ============================================================
# Stage C: Burst Replication (packing)
# ============================================================

def burst_replication(state: SystemState, v_j: int):
    r_vj = state.r(v_j)
    b_tgt: Optional[int] = None

    # prefer ON burst, lowest index, not storing v_j, with capacity
    for srv in sorted(state.bursts, key=lambda s: s.index):
        if srv.is_on and (v_j not in srv.store) and state.H(srv.index) >= r_vj:
            b_tgt = srv.index
            break

    # else turn on lowest-index OFF burst
    if b_tgt is None:
        for srv in sorted(state.bursts, key=lambda s: s.index):
            if not srv.is_on:
                state.turn_on(srv.index)
                b_tgt = srv.index
                break

    if b_tgt is None:
        return

    if v_j not in state.servers[b_tgt].store:
        ra = state.RA.get(v_j, set())
        a_src = next(iter(ra), None)
        if a_src is not None:
            state.transmit_replica(v_j, a_src, b_tgt)
        state.servers[b_tgt].store.add(v_j)
        state.RB.setdefault(v_j, set()).add(b_tgt)


# ============================================================
# De-replication (D2/D3)
# ============================================================

def de_replication(state: SystemState, v_j: int, thT: float, m: float):
    while True:
        ra = state.RA.get(v_j, set())
        rb = state.RB.get(v_j, set())

        ac = sum(state.H(a) for a in ra) + sum(
            state.H(b) for b in rb if state.servers[b].is_on
        )
        if ac < thT + m:
            break
        if len(ra) + len(rb) <= 1:
            break

        if len(rb) > 0:
            src = max(rb)
            if not migrate_all_sessions(state, v_j, src):
                break
            state.delete_replica(v_j, src)
            state.servers[src].store.discard(v_j)
            state.RB[v_j].discard(src)
        else:
            if len(ra) <= 1:
                break
            src = max(ra, key=lambda a: state.U(a))
            if not migrate_all_sessions(state, v_j, src):
                break
            state.delete_replica(v_j, src)
            state.servers[src].store.discard(v_j)
            state.RA[v_j].discard(src)


# ============================================================
# O1-style migration: find target for one stream of video v
# ============================================================

def find_target_o1(state: SystemState, v: int, src: int) -> Optional[int]:
    rv = state.r(v)

    # (1) anchor with min load
    a_tgt: Optional[int] = None
    best_load = math.inf
    for a in state.RA.get(v, set()):
        if state.H(a) >= rv and state.U(a) < best_load:
            best_load = state.U(a)
            a_tgt = a
    if a_tgt is not None:
        return a_tgt

    # (2) lowest-index ON burst (excluding src)
    for b in sorted(state.RB.get(v, set())):
        if b == src:
            continue
        if state.servers[b].is_on and state.H(b) >= rv:
            return b

    return None


def migrate_all_sessions(state: SystemState, v: int, src: int) -> bool:
    while True:
        s = state.pick_one_active_session(v, src)
        if s is None:
            break
        tgt = find_target_o1(state, v, src)
        if tgt is None:
            return False
        state.migrate_session(s, src, tgt)
    return True


# ============================================================
# Admit / Steer helpers
# ============================================================

def select_anchor_for_request(state: SystemState, v_j: int) -> Optional[int]:
    best: Optional[int] = None
    for a in state.RA.get(v_j, set()):
        if state.H(a) >= state.r(v_j):
            if (
                best is None
                or state.H(a) > state.H(best)
                or (state.H(a) == state.H(best) and state.idx(a) < state.idx(best))
            ):
                best = a
    return best


def select_burst_for_request(state: SystemState, v_j: int) -> Optional[int]:
    for b in sorted(state.RB.get(v_j, set())):
        if state.servers[b].is_on and state.H(b) >= state.r(v_j):
            return b
    return None


# ============================================================
# Burst OFF (highest index first) with O1 feasibility
# ============================================================

def get_highest_index_on_burst(state: SystemState) -> Optional[int]:
    b_highest: Optional[int] = None
    for srv in state.bursts:
        if srv.is_on:
            if b_highest is None or srv.index > b_highest:
                b_highest = srv.index
    return b_highest


def turn_off_burst_if_o1_feasible(state: SystemState, src_b: int):
    if not can_migrate_all_streams_o1(state, src_b):
        return

    # commit migrations
    videos_on_src = {
        s.video for s in state.sessions if s.server == src_b
    }
    for v in videos_on_src:
        while state.N(v, src_b) > 0:
            s = state.pick_one_active_session(v, src_b)
            if s is None:
                break
            tgt = find_target_o1(state, v, src_b)
            if tgt is None:
                return  # should not happen
            state.migrate_session(s, src_b, tgt)

    state.turn_off(src_b)


def can_migrate_all_streams_o1(state: SystemState, src_b: int) -> bool:
    # temporary headroom map
    h_tmp: Dict[int, float] = {idx: state.H(idx) for idx in state.servers}

    videos_on_src = {
        s.video for s in state.sessions if s.server == src_b
    }

    for v in videos_on_src:
        k = state.N(v, src_b)
        rv = state.r(v)

        for _ in range(k):
            # (1) anchor min-load
            a_tgt: Optional[int] = None
            best_load = math.inf
            for a in state.RA.get(v, set()):
                if h_tmp[a] >= rv and state.U(a) < best_load:
                    best_load = state.U(a)
                    a_tgt = a

            if a_tgt is not None:
                h_tmp[a_tgt] -= rv
                h_tmp[src_b] += rv
                continue

            # (2) burst lowest-index ON (excluding src_b)
            b_tgt: Optional[int] = None
            for b in sorted(state.RB.get(v, set())):
                if b == src_b:
                    continue
                if state.servers[b].is_on and h_tmp[b] >= rv:
                    b_tgt = b
                    break

            if b_tgt is not None:
                h_tmp[b_tgt] -= rv
                h_tmp[src_b] += rv
                continue

            return False

    return True


# ------------------------------------------------------------------ #
#  Smoke test                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Two anchors (idx 1,2), two bursts (idx 3,4)
    a1 = Server(index=1, capacity=100.0, is_anchor=True, store={0, 1})
    a2 = Server(index=2, capacity=100.0, is_anchor=True, store={0})
    b1 = Server(index=3, capacity=100.0, is_anchor=False, is_on=False, store={1})
    b2 = Server(index=4, capacity=100.0, is_anchor=False, is_on=False, store=set())

    bitrates = {0: 10.0, 1: 15.0, 2: 20.0}

    state = SystemState(
        anchors=[a1, a2],
        bursts=[b1, b2],
        bitrates=bitrates,
        margin=5.0,
        off_threshold=0.15,
    )

    # Set thresholds so scaling can trigger
    state.thA[0] = 50.0
    state.thT[0] = 120.0
    state.thA[1] = 40.0
    state.thT[1] = 100.0
    state.thA[2] = 30.0
    state.thT[2] = 80.0

    # Request video 0
    result = joint_policy(state, 0, start_time=0.0)
    print(f"Request video 0 -> {'ACCEPT' if result else 'REJECT'}")

    # Request video 1
    result = joint_policy(state, 1, start_time=1.0)
    print(f"Request video 1 -> {'ACCEPT' if result else 'REJECT'}")

    # Request video 2 (not stored anywhere yet – expect REJECT unless replication kicks in)
    result = joint_policy(state, 2, start_time=2.0)
    print(f"Request video 2 -> {'ACCEPT' if result else 'REJECT'}")

    # Show final state
    for s in state.anchors + state.bursts:
        tag = "Anchor" if s.is_anchor else f"Burst({'ON' if s.is_on else 'OFF'})"
        print(
            f"  {tag} {s.index}: store={s.store}, "
            f"U={state.U(s.index):.1f}, H={state.H(s.index):.1f}"
        )
    print(f"  Active sessions: {len(state.sessions)}")
    for sess in state.sessions:
        print(f"    sid={sess.session_id} video={sess.video} server={sess.server} rate={sess.bitrate}")
