"""
Custom Lightning callbacks.

Currently minimal -- norm-ratio logging is handled inside
``LipsLightningModule`` since it is tightly coupled to projection timing.

Add specialised callbacks here as needed (e.g. periodic Lipschitz-bound
snapshots, adversarial evaluation callbacks, etc.).
"""
