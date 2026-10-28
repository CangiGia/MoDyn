"""Contact and collision — **architectural placeholder**.

Reserved for future implementation. The intended layout is:

- ``geometry.py``  — collision geometry (Sphere, Box, Capsule, Mesh).
- ``detection.py`` — broad-phase + narrow-phase contact detection.
- ``models.py``    — normal force models (Hertz, Hunt-Crossley) and
                     friction models (Coulomb, regularized).

When implemented, contact forces are added to the ``Q_e`` vector via the
standard ``forces/`` interface so the solver requires no special-casing.
"""
