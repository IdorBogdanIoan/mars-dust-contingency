"""
Onboard Martian Dust Storm Contingency Decision Engine
Reference prototype implementing dynamic VO trust weighting (from IR/lidar tau),
multi-modal storm confidence fusion (lidar + 9 um IR), moving-front trajectory
checks, and wind-adjusted reachability.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import List, Optional, Tuple


class MissionType(Enum):
    TRANSIT_X_TO_Y = auto()
    SURVEY_X_TO_X = auto()


class ContingencyAction(Enum):
    MONITOR_AND_CLIMB = auto()
    ALTITUDE_BYPASS = auto()
    LATERAL_DETOUR_TO_Y = auto()
    ABORT_AND_RETURN_TO_X = auto()
    RELAUNCHABLE_SLZ_LANDING = auto()
    SURVIVAL_STATIC_LANDING = auto()
    UNRECOVERABLE_LAST_GASP = auto()


@dataclass(frozen=True)
class Vec2D:
    x_km: float
    y_km: float

    def distance_to(self, other: "Vec2D") -> float:
        return math.hypot(other.x_km - self.x_km, other.y_km - self.y_km)

    def unit_vector_to(self, other: "Vec2D") -> Tuple[float, float]:
        dist = self.distance_to(other)
        if dist < 1e-6:
            return (0.0, 0.0)
        return ((other.x_km - self.x_km) / dist, (other.y_km - self.y_km) / dist)


@dataclass
class AircraftState:
    position: Vec2D
    base_x: Vec2D
    destination_y: Optional[Vec2D]
    mission_type: MissionType
    soc: float                      # Battery State of Charge [0.0, 1.0]
    battery_capacity_wh: float      # Total pack capacity (Wh)
    v_cruise_kmh: float             # Nominal cruise airspeed (km/h)
    p_cruise_w: float               # Nominal cruise power draw (W)
    e_landing_wh: float             # Final VTOL landing energy (Wh)
    e_takeoff_wh: float             # Subsequent takeoff energy (Wh)
    wind_vec_kmh: Tuple[float, float]
    v_min_ground_kmh: float = 25.0
    max_ceiling_km: float = 4.5
    reserved_soc_floor: float = 0.15
    energy_safety_margin: float = 1.15


@dataclass
class StormEstimate:
    center: Vec2D
    radius_km: float
    velocity_kmh: Tuple[float, float]
    optical_depth_tau: float        # Estimated from 9 um IR & lidar backscatter
    top_altitude_km: float          # Measured via atmospheric lidar
    lidar_backscatter_score: float  # Normalized lidar dust backscatter [0.0, 1.0]
    scan_count: int                 # Consecutive 10s lidar/IR scans confirming anomaly
    ir_gradient_score: float        # Normalized 9 um thermal IR gradient [0.0, 1.0]

    @property
    def confidence(self) -> float:
        """Multi-modal storm confidence score C_storm in [0.0, 1.0]."""
        s_persist = min(1.0, self.scan_count / 3.0)
        return round(
            0.45 * self.lidar_backscatter_score
            + 0.35 * s_persist
            + 0.20 * self.ir_gradient_score,
            4,
        )

    def center_at(self, time_h: float) -> Vec2D:
        return Vec2D(
            self.center.x_km + self.velocity_kmh[0] * time_h,
            self.center.y_km + self.velocity_kmh[1] * time_h,
        )

    def will_cover_point(self, pt: Vec2D, time_h: float, buf_km: float = 8.0) -> bool:
        future_center = self.center_at(time_h)
        return pt.distance_to(future_center) <= (self.radius_km + buf_km)


@dataclass
class SafeLandingZone:
    slz_id: str
    position: Vec2D
    terrain_risk_score: float       # [0.0 = flat plain, 1.0 = hazardous slope]


@dataclass
class DecisionReport:
    action: ContingencyAction
    target_coords: Optional[Vec2D]
    selected_slz_id: Optional[str]
    estimated_energy_cost_wh: float
    remaining_usable_wh: float
    rationale: str


def compute_vo_trust_weight(
    feature_quality: float,
    estimated_tau: float,
    tau_blind: float = 1.5,
) -> float:
    """Computes dynamic visual odometry trust weight w_VO in [0.0, 1.0]."""
    raw = feature_quality * max(0.0, 1.0 - (estimated_tau / tau_blind))
    return max(0.0, min(1.0, raw))


def compute_usable_energy_wh(state: AircraftState) -> float:
    """Usable propulsion energy above the hardware-enforced 15% thermal floor."""
    return state.battery_capacity_wh * max(0.0, state.soc - state.reserved_soc_floor)


def estimate_leg_energy_wh(
    start: Vec2D,
    end: Vec2D,
    state: AircraftState,
    path_dust_tau: float = 0.2,
) -> Tuple[float, float]:
    """
    Returns (energy_wh, travel_time_hours) adjusted for wind, air density
    (only if path traverses dust with tau > 1.0), and the +15% safety margin.
    """
    dist_km = start.distance_to(end)
    if dist_km < 1e-3:
        return (0.0, 0.0)

    ux, uy = start.unit_vector_to(end)
    wind_along = state.wind_vec_kmh[0] * ux + state.wind_vec_kmh[1] * uy
    v_ground = max(state.v_min_ground_kmh, state.v_cruise_kmh + wind_along)

    time_h = dist_km / v_ground
    density_penalty = (1.0 + 0.12 * path_dust_tau) if path_dust_tau > 1.0 else 1.0
    energy_wh = (
        state.p_cruise_w * density_penalty * time_h * state.energy_safety_margin
    )
    return (energy_wh, time_h)


def select_detour_waypoint(
    state: AircraftState,
    storm: StormEstimate,
) -> Optional[Vec2D]:
    """
    Evaluates left (north) and right (south) flank waypoints clearing the storm
    by 12 km, rejecting any flank overtaken by the moving storm front.
    """
    clearance = storm.radius_km + 12.0
    ux, uy = state.position.unit_vector_to(storm.center)
    candidates = [
        Vec2D(storm.center.x_km - uy * clearance, storm.center.y_km + ux * clearance),
        Vec2D(storm.center.x_km + uy * clearance, storm.center.y_km - ux * clearance),
    ]

    feasible: List[Tuple[float, Vec2D]] = []
    for wp in candidates:
        e1, t1 = estimate_leg_energy_wh(state.position, wp, state, path_dust_tau=0.2)
        if not storm.will_cover_point(wp, t1, buf_km=8.0):
            e2, _ = estimate_leg_energy_wh(
                wp, state.destination_y, state, path_dust_tau=0.2  # type: ignore[arg-type]
            )
            feasible.append((e1 + e2, wp))

    if not feasible:
        return None
    feasible.sort(key=lambda item: item[0])
    return feasible[0][1]


def evaluate_contingency(
    state: AircraftState,
    storm: StormEstimate,
    candidate_slzs: List[SafeLandingZone],
) -> DecisionReport:
    usable_wh = compute_usable_energy_wh(state)

    # Tier 0: Filter transient dust devils or low-confidence anomalies
    if storm.confidence < 0.60:
        return DecisionReport(
            ContingencyAction.MONITOR_AND_CLIMB,
            state.destination_y,
            None,
            0.0,
            round(usable_wh, 2),
            f"Low confidence ({storm.confidence:.2f} < 0.60); stepping up +300 m.",
        )

    # Tier 1: Altitude bypass for shallow local dust fronts only (never RDS/GDE)
    if storm.optical_depth_tau < 1.5 and (storm.top_altitude_km + 0.5) <= state.max_ceiling_km:
        return DecisionReport(
            ContingencyAction.ALTITUDE_BYPASS,
            state.destination_y,
            None,
            8.0,
            round(usable_wh - 8.0, 2),
            "Shallow local dust front; climbing 500 m above dust ceiling.",
        )

    # Tier 2: Lateral detour for Point-to-Point Transit (X -> Y)
    if state.mission_type == MissionType.TRANSIT_X_TO_Y and state.destination_y is not None:
        wp = select_detour_waypoint(state, storm)
        if wp is not None:
            e1, t1 = estimate_leg_energy_wh(state.position, wp, state, path_dust_tau=0.2)
            e2, t2 = estimate_leg_energy_wh(wp, state.destination_y, state, path_dust_tau=0.2)
            detour_wh = e1 + e2 + state.e_landing_wh

            if (
                not storm.will_cover_point(state.destination_y, t1 + t2)
                and detour_wh <= usable_wh
            ):
                return DecisionReport(
                    ContingencyAction.LATERAL_DETOUR_TO_Y,
                    wp,
                    None,
                    round(detour_wh, 2),
                    round(usable_wh - detour_wh, 2),
                    (
                        f"Destination Y clear; detour via {wp} "
                        f"(Leg1={e1:.2f} Wh, Leg2={e2:.2f} Wh, Land={state.e_landing_wh:.2f} Wh)."
                    ),
                )

    # Tier 3: Abort mission and return to Base X (before wind-adjusted PNR)
    e_ret, t_ret = estimate_leg_energy_wh(
        state.position, state.base_x, state, path_dust_tau=0.2
    )
    ret_wh = e_ret + state.e_landing_wh
    if not storm.will_cover_point(state.base_x, t_ret) and ret_wh <= usable_wh:
        return DecisionReport(
            ContingencyAction.ABORT_AND_RETURN_TO_X,
            state.base_x,
            None,
            round(ret_wh, 2),
            round(usable_wh - ret_wh, 2),
            "Base X reachable within safety margin; executing PNR return.",
        )

    # Filter SLZs outside storm footprint at touchdown, ranked by energy + terrain risk
    valid_slzs: List[Tuple[float, SafeLandingZone]] = []
    for slz in candidate_slzs:
        e_slz, t_slz = estimate_leg_energy_wh(
            state.position, slz.position, state, path_dust_tau=0.2
        )
        if not storm.will_cover_point(slz.position, t_slz):
            valid_slzs.append((e_slz + 40.0 * slz.terrain_risk_score, slz))
    valid_slzs.sort(key=lambda item: item[0])

    # Tier 4: Re-launchable Safe Landing (energy covers landing AND later takeoff)
    for _, slz in valid_slzs:
        e_slz, _ = estimate_leg_energy_wh(
            state.position, slz.position, state, path_dust_tau=0.2
        )
        relaunch_wh = e_slz + state.e_landing_wh + state.e_takeoff_wh
        if relaunch_wh <= usable_wh:
            return DecisionReport(
                ContingencyAction.RELAUNCHABLE_SLZ_LANDING,
                slz.position,
                slz.slz_id,
                round(relaunch_wh, 2),
                round(usable_wh - relaunch_wh, 2),
                (
                    f"Landing at {slz.slz_id} (Transit={e_slz:.2f} Wh, "
                    f"Land={state.e_landing_wh:.2f} Wh, Reserved Takeoff={state.e_takeoff_wh:.2f} Wh)."
                ),
            )

    # Tier 5: Survival-Only Static Landing (protects hardware; waits for solar recharge)
    for _, slz in valid_slzs:
        e_slz, _ = estimate_leg_energy_wh(
            state.position, slz.position, state, path_dust_tau=0.2
        )
        survival_wh = e_slz + state.e_landing_wh
        if survival_wh <= usable_wh:
            return DecisionReport(
                ContingencyAction.SURVIVAL_STATIC_LANDING,
                slz.position,
                slz.slz_id,
                round(survival_wh, 2),
                round(usable_wh - survival_wh, 2),
                f"One-way survival landing at {slz.slz_id}; 15% heater reserve intact.",
            )

    # Tier 6: Unrecoverable contingency -> Black-box flush & UHF orbital burst
    return DecisionReport(
        ContingencyAction.UNRECOVERABLE_LAST_GASP,
        None,
        None,
        round(usable_wh, 2),
        0.0,
        "No SLZ reachable; initiating minimum-sink glide and UHF orbital burst.",
    )


if __name__ == "__main__":
    BASE_PARAMS = dict(
        battery_capacity_wh=350.0,
        v_cruise_kmh=100.0,
        p_cruise_w=120.0,
        e_landing_wh=18.0,
        e_takeoff_wh=45.0,
    )

    # Scenario A: Transit X -> Y, 80% SOC, leading edge 20 km ahead, drifting NW
    state_a = AircraftState(
        position=Vec2D(20.0, 0.0),
        base_x=Vec2D(0.0, 0.0),
        destination_y=Vec2D(120.0, 0.0),
        mission_type=MissionType.TRANSIT_X_TO_Y,
        soc=0.80,
        wind_vec_kmh=(-10.0, 0.0),
        **BASE_PARAMS,
    )
    storm_a = StormEstimate(
        center=Vec2D(55.0, 0.0),
        radius_km=15.0,
        velocity_kmh=(-18.0, 24.0),
        optical_depth_tau=2.4,
        top_altitude_km=6.5,
        lidar_backscatter_score=0.90,
        scan_count=3,
        ir_gradient_score=0.625,  # C_storm = 0.88
    )
    slzs_a = [SafeLandingZone("SLZ-1", Vec2D(30.0, -25.0), terrain_risk_score=0.10)]

    # Scenario B: Survey X -> X, 55% SOC, 25 km/h return headwind (past PNR)
    state_b = AircraftState(
        position=Vec2D(70.0, 0.0),
        base_x=Vec2D(0.0, 0.0),
        destination_y=None,
        mission_type=MissionType.SURVEY_X_TO_X,
        soc=0.55,
        wind_vec_kmh=(25.0, 0.0),
        **BASE_PARAMS,
    )
    storm_b = StormEstimate(
        center=Vec2D(110.0, 0.0),
        radius_km=20.0,
        velocity_kmh=(-25.0, 0.0),
        optical_depth_tau=2.4,
        top_altitude_km=6.5,
        lidar_backscatter_score=0.90,
        scan_count=3,
        ir_gradient_score=0.625,
    )
    slzs_b = [SafeLandingZone("SLZ-2", Vec2D(55.0, -20.0), terrain_risk_score=0.15)]

    # Scenario C: Transit X -> Y, 30% SOC, Y engulfed, survival-only landing
    state_c = AircraftState(
        position=Vec2D(80.0, 0.0),
        base_x=Vec2D(0.0, 0.0),
        destination_y=Vec2D(120.0, 0.0),
        mission_type=MissionType.TRANSIT_X_TO_Y,
        soc=0.30,
        wind_vec_kmh=(0.0, 0.0),
        **BASE_PARAMS,
    )
    storm_c = StormEstimate(
        center=Vec2D(120.0, 0.0),
        radius_km=20.0,
        velocity_kmh=(-15.0, 0.0),
        optical_depth_tau=2.4,
        top_altitude_km=6.5,
        lidar_backscatter_score=0.90,
        scan_count=3,
        ir_gradient_score=0.625,
    )
    slzs_c = [SafeLandingZone("SLZ-2", Vec2D(75.0, -15.0), terrain_risk_score=0.15)]

    for label, st, stm, slzs in [
        ("Scenario A (Transit Detour)", state_a, storm_a, slzs_a),
        ("Scenario B (Survey Past PNR -> Relaunchable SLZ)", state_b, storm_b, slzs_b),
        ("Scenario C (Low SOC -> Survival SLZ)", state_c, storm_c, slzs_c),
    ]:
        report = evaluate_contingency(st, stm, slzs)
        print(f"=== {label} ===")
        print(f"  C_storm:   {stm.confidence}")
        print(f"  Action:    {report.action.name}")
        print(f"  Target:    {report.selected_slz_id or report.target_coords}")
        print(f"  Budget:    {report.estimated_energy_cost_wh} Wh (Remaining usable: {report.remaining_usable_wh} Wh)")
        print(f"  Rationale: {report.rationale}\n")