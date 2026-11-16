"""Separate robot and simulation histories while sharing one serialized model.

This is request routing, not a robot watchdog. Each scope has one replaceable
binding, so recovery does not accumulate abandoned image histories.
"""


class InferenceSessions:
    def __init__(self, factory):
        self.factory = factory
        self.sessions = {}

    @staticmethod
    def scope(packet):
        if not isinstance(packet, dict):
            raise ValueError("Expected a JSON object")
        # Older simulation adapters already use this prefix. Keep them from
        # replacing an operator binding even before their code is restarted.
        binding = packet.get("binding_id", "")
        legacy_simulation = isinstance(binding, str) and binding.startswith("simulation-")
        scope = packet.get("client_scope", "simulation" if legacy_simulation else "operator")
        if scope not in ("operator", "simulation"):
            raise ValueError("client_scope must be operator or simulation")
        if legacy_simulation and scope != "simulation":
            raise ValueError("Simulation bindings must use the simulation scope")
        return scope

    def request(self, packet):
        scope = self.scope(packet)
        session = self.sessions.get(scope)
        if packet.get("op") == "reset":
            if session is None:
                session = self.sessions[scope] = self.factory()
        elif session is None or session.binding is None or packet.get("binding_id") != session.binding:
            # A delayed stop/observation from an old client must not clear the
            # new binding in this scope, or any binding in the other scope.
            raise ValueError(f"Reset required or binding mismatch in {scope} scope")
        try:
            return session.request(packet)
        except Exception:
            # Preserve the existing invalidation rule only for this owner.
            session.stop()
            raise

    def stop(self):
        for session in self.sessions.values():
            session.stop()
