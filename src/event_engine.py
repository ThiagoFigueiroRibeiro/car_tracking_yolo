class EventEngine:
    def __init__(self):
        self.active_ids = set()

    def process(self, new_tracks, removed_tracks):
        events = []

        for t in new_tracks:
            self.active_ids.add(t.track_id)
            events.append({
                "type": "enter",
                "track_id": t.track_id,
                "class_name": t.class_name,
                "first_seen": t.first_seen,
            })

        for t in removed_tracks:
            self.active_ids.discard(t.track_id)
            events.append({
                "type": "exit",
                "track_id": t.track_id,
                "class_name": t.class_name,
                "first_seen": t.first_seen,
                "last_seen": t.last_seen,
                "duration_sec": max(0.0, t.last_seen - t.first_seen),
            })

        return events