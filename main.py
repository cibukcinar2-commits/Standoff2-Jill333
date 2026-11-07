import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Deque
from collections import deque

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Vector3') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class PositionSample:
    timestamp: float
    position: Vector3

@dataclass
class PlayerState:
    player_id: str
    position: Vector3
    last_update_time: float
    max_speed: float = 7.0  # saniyedeki birim
    # Gecikme telafisi ve geçmiş takibi için son pozisyon örnekleri (ör. son 1 saniye)
    history: Deque[PositionSample] = field(default_factory=lambda: deque(maxlen=60))

class ServerMovementValidator:
    """
    Hız manipülasyonunu, ışınlanmayı ve geçersiz konum durumlarını önleyen
    gelişmiş sunucu yetki yöneticisi.
    """
    def __init__(self):
        self._player_states: Dict[str, PlayerState] = {}

    def register_player(self, player_id: str, initial_pos: Vector3, timestamp: float):
        state = PlayerState(
            player_id=player_id,
            position=initial_pos,
            last_update_time=timestamp
        )
        state.history.append(PositionSample(timestamp, initial_pos))
        self._player_states[player_id] = state

    def validate_and_update_movement(self, player_id: str, new_pos: Vector3, timestamp: float) -> bool:
        if player_id not in self._player_states:
            return False

        state = self._player_states[player_id]
        delta_time = timestamp - state.last_update_time

        if delta_time <= 0:
            return False

        distance_moved = state.position.distance_to(new_pos)
        max_allowed_distance = state.max_speed * delta_time * 1.15  # %15 tolerans payı (jitter için)

        if distance_moved > max_allowed_distance:
            print(f"[REJECTED] Oyuncu {player_id} sınırı aştı! Hareket: {distance_moved:.2f}u, Max: {max_allowed_distance:.2f}u")
            return False

        # Durumu güncelle ve geçmiş tamponuna ekle
        state.position = new_pos
        state.last_update_time = timestamp
        state.history.append(PositionSample(timestamp, new_pos))
        print(f"[ACCEPTED] Oyuncu {player_id} konumu güncellendi: ({new_pos.x:.1f}, {new_pos.y:.1f}, {new_pos.z:.1f})")
        return True

    def get_position_at_time(self, player_id: str, target_timestamp: float) -> Vector3 | None:
        """Gecikme telafisi (Lag Compensation) için geçmişteki bir zamana ait pozisyonu hesaplar (Lineer İnterpolasyon)."""
        if player_id not in self._player_states:
            return None
        
        history = self._player_states[player_id].history
        if not history:
            return None

        # Eğer talep edilen zaman çok eskiyse en eski kaydı döndür
        if target_timestamp <= history[0].timestamp:
            return history[0].position

        # Eğer talep edilen zaman mevcut zamandan ilerideyse güncel pozisyonu döndür
        if target_timestamp >= history[-1].timestamp:
            return history[-1].position

        # Zaman aralığındaki iki örneği bul ve interpole et
        for i in range(len(history) - 1):
            s1 = history[i]
            s2 = history[i+1]
            if s1.timestamp <= target_timestamp <= s2.timestamp:
                factor = (target_timestamp - s1.timestamp) / (s2.timestamp - s1.timestamp)
                # HATA DÜZELTMESİ: s1 ve s2 PositionSample sınıfıdır, pozisyon .position ile alınır.
                return Vector3(
                    x=s1.position.x + (s2.position.x - s1.position.x) * factor,
                    y=s1.position.y + (s2.position.y - s1.position.y) * factor,
                    z=s1.position.z + (s2.position.z - s1.position.z) * factor
                )
        return history[-1].position

class ServerCombatValidator:
    """
    Duvar arkasından vurma (wallbang/noclip check) ve isabet doğrulaması için Slab Metodu raycasting.
    """
    @staticmethod
    def is_line_of_sight_blocked(origin: Vector3, target: Vector3, obstacles: List[Tuple[Vector3, Vector3]]) -> bool:
        for min_bound, max_bound in obstacles:
            if ServerCombatValidator._ray_intersects_aabb(origin, target, min_bound, max_bound):
                return True
        return False

    @staticmethod
    def _ray_intersects_aabb(origin: Vector3, target: Vector3, min_b: Vector3, max_b: Vector3) -> bool:
        dir_x = target.x - origin.x
        dir_y = target.y - origin.y
        dir_z = target.z - origin.z

        tmin, tmax = 0.0, 1.0

        # X Ekseni
        if abs(dir_x) < 1e-6:
            if origin.x < min_b.x or origin.x > max_b.x:
                return False
        else:
            inv_d = 1.0 / dir_x
            t1 = (min_b.x - origin.x) * inv_d
            t2 = (max_b.x - origin.x) * inv_d
            if t1 > t2: t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax: return False

        # Y Ekseni
        if abs(dir_y) < 1e-6:
            if origin.y < min_b.y or origin.y > max_b.y:
                return False
        else:
            inv_d = 1.0 / dir_y
            t1 = (min_b.y - origin.y) * inv_d
            t2 = (max_b.y - origin.y) * inv_d
            if t1 > t2: t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax: return False

        # Z Ekseni
        if abs(dir_z) < 1e-6:
            if origin.z < min_b.z or origin.z > max_b.z:
                return False
        else:
            inv_d = 1.0 / dir_z
            t1 = (min_b.z - origin.z) * inv_d
            t2 = (max_b.z - origin.z) * inv_d
            if t1 > t2: t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax: return False

        return True

if __name__ == "__main__":
    validator = ServerMovementValidator()
    validator.register_player("P_01", Vector3(0.0, 0.0, 0.0), timestamp=10.0)

    # Geçerli hareketler
    validator.validate_and_update_movement("P_01", Vector3(3.0, 0.0, 0.0), timestamp=10.5)
    validator.validate_and_update_movement("P_01", Vector3(6.0, 0.0, 0.0), timestamp=11.0)

    # Geçmiş tamponundan (lag compensation) belirli bir andaki pozisyonu sorgulama testi
    past_pos = validator.get_position_at_time("P_01", target_timestamp=10.75)
    if past_pos:
        print(f"[LAG COMP] P_01'in 10.75 anındaki interpolasyon pozisyonu: ({past_pos.x:.2f}, {past_pos.y:.2f}, {past_pos.z:.2f})")