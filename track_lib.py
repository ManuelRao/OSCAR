import json

def lerp(p1, p2, alpha):
    return (p1[0] + alpha * (p2[0] - p1[0]),
            p1[1] + alpha * (p2[1] - p1[1]))

class Track:
    def __init__(self, track_file="track.json"):
        with open(track_file, "r") as f:
            self.data = json.load(f)

        self.points = [tuple(p) for p in self.data["points"]]
        self.segments = [(tuple(a), tuple(b)) for a, b in self.data["segments"]]
        self.lengths = self.data["lengths"]
        self.total_length = self.data["total_length"]
        self.cumulative = self.data["cumulative"]

    def get_position(self, t: float):
        """Get position on track at normalized progression t ∈ [0,1]."""
        t = t % 1.0  # wrap around loop

        # Find segment where t lies
        for i in range(len(self.segments)):
            t0 = self.cumulative[i]
            t1 = self.cumulative[i+1]
            if t0 <= t < t1:
                p1, p2 = self.segments[i]
                local_alpha = (t - t0) / (t1 - t0)  # local interpolation factor
                return lerp(p1, p2, local_alpha)

        # Edge case: exactly at the end
        return self.points[0]


if __name__ == "__main__":
    track = Track()
    for test_t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pos = track.get_position(test_t)
        print(f"t={test_t:.2f} → position={pos}")
