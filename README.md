# MoDyn

**MoDyn** is a Python library for the dynamic simulation of 3D rigid multibody
systems. It is the spatial counterpart of [PMD](https://github.com/CangiGia/PMD):
the same formulation, based on bodies, markers and joints, is extended to
translations and rotations in space, with orientation described by Euler
parameters (unit quaternions).

> **Status: work in progress.** The API is not stable and parts of the package
> are still being developed and validated.

## Install (development)

```powershell
pip install -e .[dev]
```

## Layout

```
src/modyn/      library
tests/          unit, integration and regression tests
validation/     comparisons with PMD and ADAMS
docs/           documentation sources
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design notes.

## License

See [LICENSE](LICENSE).
