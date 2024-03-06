# `sirf.STIR.AcquisitionData` Partioner

This folder contains a function to split `sirf.STIR.AcquisitionData`
into subsets, as well as create acquisition models and subsets
that use this data.

It is envisaged this will be moved to SIRF itself, after some suitable
improvements.

A lot of the functionality and code is identical to
[CIL.DataPartioner](https://github.com/TomographicImaging/CIL/blob/b20cc5679a56e26bc3e41bca2497b09cf27efe3c/Wrappers/Python/cil/framework/framework.py#L48).
However, `sirf.STIR` needs more complicated construction of `AcquisitionModel`s
than CIL.
