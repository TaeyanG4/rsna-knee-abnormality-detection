"""HB126 codecs v1.0.0: offline python-gdcm for every interpreter of a release (H-B-126 step 2).

The pinned Kaggle image (pydicom 3.0.2; the S6/R7 stage installs pydicom 3.0.1) has no decoder for
JPEG Lossless (.4.57/.4.70) or JPEG-LS (.4.80/.4.81) (D-B-162). This module installs the pinned
python-gdcm cp313 wheel from a private dataset into a fresh private directory (``pip install
--no-index --no-deps --target``; safe unzip as fallback) and writes a hook directory next to it: the hook module
``knee_hb126_hook.py`` plus one start-up entry, ``usercustomize.py`` (preferred) or, when the user
site is disabled, ``sitecustomize.py``; either entry imports the hook and then executes the next
module of the same name on sys.path (chaining, so an image's own customize module still runs). A
child interpreter whose PYTHONPATH contains the hook directory then, at start-up:

* appends the private gdcm directory to ``sys.path`` (gdcm is importable only where the hook ran);
* arms a one-shot import finder that, once ``pydicom.pixels.decoders.base`` has executed, moves
  the ``gdcm`` plugin to the END of every decoder that already had another available plugin
  (Pillow J2K/JPEG, pylibjpeg, ...): whatever decoded before keeps its plugin, GDCM is only a last
  resort there, and it is the only decoder for JPEG Lossless / JPEG-LS;
* wraps every compressed-syntax plugin to count successes and failures per branch
  (``KNEE_CODEC_BRANCH``) into per-process JSON files under ``KNEE_CODEC_TELEMETRY``.

Uncompressed transfer syntaxes never reach a plugin, so their decoding is untouched. Every step is
guarded; install() never raises and returns a receipt whose status is ``ACTIVE`` only after a
fresh interpreter proved: hook loaded and patched, gdcm imported, the plugin orders above, and
exact decodes of four embedded synthetic frames (.57, .70, .80 and .90). Anything else returns an
``INACTIVE_*`` status and callers keep their environment unchanged (the pre-codec behaviour).

Changelog v1.0.0 (2026-10-13, Claude opus builder stand-in for B): first version.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

VERSION = "1.0.0"
DATASET_SLUG = "taeyangg4/knee-hb126-codecs-cp313-v1"
MANIFEST = "hb126-codecs-manifest.json"
WHEEL = {
    "file": "python_gdcm-3.2.6-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    "size": 13277668,
    "sha256": "69c987d3f1b3a2a9936288f312f1ea2192e39fa6c0be28159b06a1906b8d56ed",
    "package": "python-gdcm", "version": "3.2.6", "licence": "Apache-2.0 (PyPI license_expression)",
    "url": "https://files.pythonhosted.org/packages/30/ba/52fa278ff973ae2932ffdde98e9bdeaa248abda39c1c74e39caa969d543e/"
           "python_gdcm-3.2.6-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
}
BRANCH_ENV = "KNEE_CODEC_BRANCH"
TELEMETRY_ENV = "KNEE_CODEC_TELEMETRY"
HOOK_MODULE = "knee_hb126_hook"
ENTRY_NAMES = ("usercustomize", "sitecustomize")  # preferred first
HOOK_VERSION = "1.0.0"
# No decoder for these in the pinned image (D-B-162): gdcm must be available for them.
TARGET_UIDS = ("1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70",
               "1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81")
SKIP_WALK = {"train_series", "test_series", "train_images", "test_images", "__pycache__"}
# Synthetic 36x44 12-bit frame (no patient data): JPEG lossless p6 / SV1 (libjpeg-turbo),
# JPEG-LS NEAR=0 (CharLS) and reversible J2K (OpenJPEG), each verified locally by an independent
# decoder (imagecodecs) to reproduce the array whose little-endian uint16 bytes hash to array_sha256.
SELFTEST = json.loads((
    '{"array_sha256":"12ebb130644c038f6cad8dadd9c663f23c8fcec2247410c05ebe489a367efda8","bits_stored":12,'
    '"columns":44,"frames":{"1.2.840.10008.1.2.4.57":"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'
    '/0CZBu9Pw/9k=","1.2.840.10008.1.2.4.70":"/9j/4AAQSkZJRgABAQAAAQABAAD/wwALDAAkACwBAREA/8QAGQAAAwEBAQA'
    'AAAAAAAAAAAAACQoMCwgH/9oACAEBAAEAAJ//ANm9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m9m'
    '9m9m9m9m9m9m9m9m9mgZfZvZu2Pe+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e+j/e4I73vo/3vo/3vo/3'
    'vo/3vo/3vo/3vtjqTpDqTrjqTpDqTrjqTpDqTrjqTpDqTrjqTpDqTrjqTpCCj646k6Q6k646k6Q6k646k6Q6k646k6Q6k646k6Q6'
    'k7Y6565657h6565657h6565657h6565657h65guOue4eueueue4eueueue4eudfLrmDg6565657h6565657h6565657Y7i7i7i66'
    '7i7i7iIV3F3F3F113F3FBwkK7i7i7i667i7i2GyFdxdxQcPXXcXcXcRCu4u4u4uuu4u4u4iFdxdxdxdsd7kL737j73IX3uTPvchf'
    'e8HF3uQvvcmfe+yJ3v3H3uQuD2Jn3uQvvfuPvchfe5M+9yF97zDfe5C+9yZ97kL737YJAQwkBDCQEMJAVwkEIWSAhhICGbLJXCQE'
    'MJBCFkgIYSArhICGEgIYSCZEJAVwkBDCQEMJAQwkBXCQEMJB2wTUmpNSak1JqTUwsJoTUmpNdm4mpNTCwmhNSak1JqTUmphSazNp'
    'NSak1JqTUwpNSak1JqTUmszaYUmpNSa9sFeK8V4rxXivQrpDivFeK9tLleK9CumuK8V4rxXivFemliHFeK8V4rxXivFeNcV6aWK8'
    'V4rxXivEOK8V6Fd7YLmYcuZhy5wwxcyblz2wC5mHLnDDFzOuXMw5c9sCFyMOXMm5czDlzMOXOa/LmdcuZhy5mHLmYeFy2cC5mHLn'
    '2wZwxBnDYQzZiDOFg21TEGchrzOGIM4wAZzbBhmzYGcMQZwsBnDETahsDOGIM4wAZwxEM3t0mcMQZwsBnIYcznbBsTYmxh2DYmxN'
    'iYnbqNjDYHZNibE2LFM3SbE2J2TYmxNiYmbpNibE7JsTYmxiKNuo2JsTsmxNjDYGJ26jYmx7YO0do7UNido7R2tus7UOydo2R2jt'
    'bicSido7R2jZHaO1OJGyO0do7RsjtTiR2mUjtHaO1DYna3EztGyO1Dsna7YPewHD3HbPewHuUHbh7mAz3nbPfOdnvZ7PewGe87c5'
    'OwGe87Z72Az3ziZ72Az3s9w9zAe5Qds97AcPcds9+54e/thhhgSIXYEYYYE3UYgNhhgRhhgSdQYEYYakYYYEYYnPGGGBGGGBGGJz'
    'xhhgRhhgSIX33WGGBGGIgNhhgTdRYEiF2BGGO2GLYixi1i1i3dmiLGLWLWLWLZ2Vi1i1i1s1i2dlYtYtYtYtYtnZWLWLWLWLYizd'
    'mYtbNYtiLGLd2Zi1i2IsYtYt3ZoizthlaJVZWZWZWnjmVmVmVmVmVp45lZlZlY208cysysysysytPHMrMrMrRKu8eysysytF6srb'
    'x7K0SqysytvHsrRKrKzK3bETUz+zWz+zXPbs1s/s1s/z0bP7NbP7Nc4uzWz+zWz/PRs/s1s/s1xP29Iz+zXE/M10LDNcT8zWz/vS'
    'RPzNbP7Ne9xE1dsRSbQDSjVE+Y0A0o1Q0pPcNKNUNKNAT5jBDSjQDSk/A0o0A0pFT75rQDSkVO0pvctKRo7SjQG+bFTtKNANKT8D'
    'SjQDSkGu1S1S1S2jPwtUtUtotUz8LVLaLVMVRvxMYtUtUz8LaLVLVMVRQFNUtUxVDaO/E1TFUO478UVQ1S2i1TPwtUtotUtUz8Pb'
    'DaTaTaTVWAU2k2k31gFNpNpNVNpYBTaTLTaWAU2k1VFotpUBbfUWi2lQFtVRaLaTaVGnFotpNpT8TaTaTaTfWAU2k2l2w3O323O2'
    'ngPt9tzuR4D7fbc7aeA+323O0HgPt9tzxaTc9Bo3PGRNz0Gjc8Wk3PQaRcz4bc9BpFzNptzt94D7kbc7feA/2w4g344g35gkN+OI'
    'RpdCS344hF90JLfjiEVXQkt+Rh7flCS35GHumUJMX24g35QkxfbiD9mCQ344g35gkN+OIOmYJDfjiHbDkrkrkuC65K5K5LhWuSuS'
    'xklC85K5LGSUBzksZI5LQvOSxkjktFdGSOSuS4LrkrktC9IKOSuS4LrkrkrkuC67q5K5Lgu9sOnOnOnYQzpzp0abQbunOnRptEO6'
    'dGm0Q7f0aa6dRDunRprp1EPF+unUQ8aa6c6dhDOnaHzp2EM6c6c6dhDOnN/YQzpzp3bDsbu7seFe7G7vGxUMDscbtROO7xsTu9E5'
    'GTOx0WDscbs7HRYRsTk9E5G7Oxu74Tbu7sejPhNu7ux4V7sbu7seC+7G7vhN9sPGO8RxdGy8ZG70YrqDxkbvRivQRxdFi8ZGnvGU'
    'WMcW9BRixu7xlETHFu8PGYbDxjvGGJpIPGO8YYj0DxmFg8Y6hHF0WLxnbD0L0MdBSBPQx0FG07zHQUbT0MeQ9DRtR0DvNG1HQPQ0'
    'gUdA9C9DhYvQvQ4bL5L0OGy9Dpc4bL0L0L5OGy9C9DhYvQvQ4bPbD5T5UeTRtvlR5NIHHQvlUgceS9FSBx5L5VG3HkvlPlYbT5T5'
    'WIE9E+ViBPlPRYgT5T5WQ0+U+ViBPRPlR5NIG9FHk0gb5XbD78fk+/SCR7z+dIxHlPv0lke8+XSMR+T7+IG++/niLvlvv4lb775e'
    'Iu/m+/iBvvv54i+oW+/iVvvvl4i7+b7+IG+/H5UjHbGgRH50oL+kgPSW6BEfnSgx+egRSWyA7+lKDH56BGJZoEP6YoD+mgRiWaBD'
    '+mKA/poEYlmgRqaYoD+mgRiWaBEfnSgv6SA9JboEdsaDsg5SraDsg5SrSDmg7SrSDmg7iq6Dug7iq6Dug7iq6Dug7iq6DuKroO6D'
    'uKroO6DuKroO6DuUroOyDlKtoOyDlKtoOyDlKtIOaDvbGiNIi0v8AIi6I1L/Ii6I2L9ojaI2L9ojYv2iNojYv2iNojYv2iNojYv2'
    'iNojYv2iNi/aI2iNi/ODSItL/AKI0iLS/yItL/ojSItL/AKI3bEi1o30ykjdot40+i3o34yejfjJ6N+i3jT6LejfjJ6N+Mno36Le'
    'NPot6N+Mno34yejfot40+i3GXUymjfItU1Ei1TUaLcjdTKSN2i32xJB6OFN7JL6QmNRpCY6GkJo4Y3mkxpCY1GkJjoaQmjhjeaTG'
    'N5o4aQmOhpCSN9N7pMY3mjhJB0SOkJI303skvTeyN+kJToyQdNTJB9sSTGkzjo6YOkzjo6TOPzpM6TOOjpg46OkzpM4/Okzjo6TO'
    'mDjo6TOOjpgyTFOlpM4/OkzJMU6TwMkxTpSTFP1JMaTNOlJgU6Ukxa+//2Q==","1.2.840.10008.1.2.4.80":"/9j/6AAgU1B'
    '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","1.2.840.1000'
    '8.1.2.4.90":"/0//UQApAAAAAAAsAAAAJAAAAAAAAAAAAAAALAAAACQAAAAAAAAAAAABCwEB/1IADAAAAAEABQQEAAH/XAATQGB'
    'oaHBoaHBoaHBoaHBoaHD/ZAAlAAFDcmVhdGVkIGJ5IE9wZW5KUEVHIHZlcnNpb24gMi41LjL/kAAKAAAAAAmcAAH/k8PzBgl9wUz'
    'RF8P2BIfsCQH4AQAAFfK/BKisHwlzw/YOj+QqH8hQA7ut1VET1iCJzaCQTh8C8crGybWloNIaCnAkZZRhdZ1xR8fyTz/BRH+CIA4'
    'S+yN5YdFZD1lE5DBID8rfPOkxPqegkJFWW7G5Ct0TYGJwM/vf7xgKZwsyRoPZ/vnia2mVHyevu3mip43tadr7KGXJXKsK0o+HRoT'
    'bfwcMl1faXj7pX2ZECqe/CTTZBVBBV/CKo/0DO1JDlU32y6vuz/EpP8TE/2AUYAP/ZII5X1GWbgWCnUnXuPuLZjoOPJw9lyvzOyw'
    'eRknLwLBDMPk7eUifKwqfMTv5fcBFhelikf9bAy0oDeiYf0TXAfNRp4pLXzA4gn8rEBTualwW8eHeUQki/kSw3REaLkzOf75m1gL'
    'fUiuq0KqL6XGezDzlYSnTbONEplrNNW4t4GORzesR0SmeFVkfC3JoP46c+X8brIqYcmKQDjBLD9UmFzzroVlMWfl1zJ7FRzHFHO7'
    'reOeyricTiCsAR3P7ny9B91+rlroccTlnVWsT0yjGRb0YgAjLhhbOj/LkI8Pb+HjI3wBYbO67IYVPBqOp1jqz9VgNhWmWm6UtD3h'
    'Rn9pzaTgM+eD1F2BnFSYScmQzPrdFI+dQM8l0aPRdc3ymkOCXgaXsyghGD8JNTx54Ij1hwuC0TNi9p1uQ9aiPCGG50RM3RxqWyEu'
    'YGS1qA3cZ3YUwBkLG5Zog0Cks8hCjiJTnKgw0+UNslr2huzpWdaIxD5IO6VRyom9TkqiSv8FukBezo6AFVpWkMz/c9XN8jsu+p3R'
    '2mNm2XPw437Ml/P1e7lOwvEFJ785jst9K2OHFw+LOtmNnsiDb2+dvYyi1xiZRri+EYZ9IH/i7LY9KdjfP80nT/NL8/2A0doL72xn'
    'SelXW6I96MzsiSNpUS8I2J8vd5Bq/IVaFccu7Cio88mCsBs7UO8R8ga0ql9gz3toXL8x28GXcCWL4CePFWIBHqXTicbBb2YtZGuT'
    'JUOIFqVxe725t/jnA7f5m/lHH/C1Agb4oyJc3oCdpvErr1t3l2KLUo0fqFXYJWxSqCoy3e5u0cCY3Lo2sMWVnaboEUf0v7MkCj5i'
    'f5TMznHPEjYl6DjbJfozwk74LZdGUKqCFzY8v2QbqybrCst/PwAMhkrqp+Vtln2JlzgpAwnK+Kv84RIYdrv3SQuDgkllJc64KTw3'
    '43dRtzin0uOcJYQywC7Yzm3A4CsCjNU6WVxaRDlyJ1mQWf/YlPryZRRM7vojbcHIP59ws1SWOvbUZNu+wH3p7FIzsFbCvcuGSazs'
    'XWRPSA9Z/hdxEJLV7IW/6Kalb9+9tzPLm8a1TsCW/Rpl+hQUEkMH9rV4znD6tvNyFtoxHeURSlNcikMmA2n7WR7JrnPor/BnsZya'
    'xOpZ0599b4U1582jxPwQUe2lh6FzJxm6YaWNLC5yes1PQQ09ft0gFDuNGFEQPVl77d+7WWBTaXC2WwVZYpL14+Qz2lGTxd4HLlRY'
    'yrBbR0dW2TLPTQx76Rt4D5sDRDiRn3tCLCgHumEyr8C0ecv3ZeEzFDMG1/4L7eCmp3sQetEg0Bu3U+wwcVyHSqtvSvyecb+Ye/d2'
    'aeIZYf1FqG6YcPBTIXEkOd64JEQ34K4iLWj69t0q0JJ3F7mS/iVWP/QSgtplLQpMX5XQA6ulOlRSW58f3LdfqP2DvXL2zQ/dbTu/'
    '+TMv+eCYwVPCMN3of6jRIKqepOqKUEpliOkNo/YGXDANccNmg9QzGP6YS93m0FGCVrlsMCCJDynNTylvxdPzH5aWV3ROu8L2Gjb1'
    'xWaxdOUZiJdEuQhqZsclBUAO/7VR2FrAaK66CMzqpoOgkdVQLdkaXp04LWTBKXIJGY1tGvS+dduih4hDfSrdzLlRgNAcIBuc3Z7v'
    'uxnh7kce/OS07PduLNhQHiSPkSa1v9MbhgHymT+gf5atIwkrar65LwqAd1XuR2nVBHArqdHQwcuiTIV+fgxa28C9mEAfyVwhgSIX'
    'RBGfRKNiLZbue9Vy84e687xfimcYYzIiTLXD41n6KBOl5So90oIsbGQDkH8al3SWEfnSnK2/I8uy1tIz8QBhHHd6i401+djdWIo8'
    'MLJe7fR5koOq6iMH344FuwyiL/cqhIOjRgCTP1kDm7MDk9GvF3tnMAp2sIDjw9CfBTIKVOe9tBDM7+IiC7V4nI4bbcG9R/yj6GEe'
    'rMH1Dw2q3xT/Mv1wJI2rCV4Pdv/RZnr93kMZc+urJNMnmtMnbAUsFwwV1aA/Qkbcw1bY1XeQStyybTSDdNSC7Qbz33F0L1zu/BIJ'
    'RklrRHKO/+EcOUmZwmINOWGI7nBlv9/Gii6VFtJ04vuWjfEp9JULFHER2P+b3mYfwiO+czGbvBfqE8uD7dwLbXkdb41z7Poq6F3b'
    'U5SB48UGAv8GJdk6W/j9sMlZQHc+XnyfJWQdXr2Pjyl5BA2+UmrNx2OPTYPxLrDToWlQA9utw5uptfhP12VwSfJ9ehZhNkY5gTNj'
    'JoXLA8t4iUeExlJdKp5t4bg2VPW/Ax7GlAbzEoCamE/sxH4Ll+XIFzZQLJDnSnD5zu6gPFNFzTtkKJzeGgd3rgDufBgoHIBlXmYr'
    'O0dUkC/0bNVssTFGXBRJ64/8V+sTAU/17f447dcIx5LV56eNAqQzo+gb8hNhSpdKN4ZdIBB76HJT2Z8COiH63w0FrE+rndFQqOTO'
    'nn/0Bbp4vhYzQgYr6PoyMbBP8edscoWtALxbK2dfvJChCAwVoCyL4kAFym5phut+NVdjIwTNJuDj57/904JZJQTclWu84HYjRyh7'
    'z6E9wZz+572PO3G3TLg4gzHroBfJ5b3sFqFgRyZJa4U+5nktPXS86H6PZVD0AQDlUI2PN2jVpkE4m9o/EX/cfB5hQMrfiohMhSX8'
    'uQp5e3UyAgNRRANGJw5dw/KtWc/lpOF4woG0co26GcPBaoojXPKcOC6fTSDUzAVcLMfdSGTsQ34s3rwcG4BlN7uLw4mp0elTs6ZO'
    'WQneTZfSQkL5O4mhiHZmpLmKlPm+eTQdOurnjdkWgrQs0Evc0s7S9TyBUJxt2iQs+cV/apW/BcpdeahpQeFZcAk7QTTD9mYe5s5u'
    '9oBFH5UW1mJHTQzWZUaZ4QeajhGUdqDja6d1PO1Axu1R0oLez2215fOh4N4ZKWn/NphR//NbOH2//arku4vlso9f4E+NMQp3qH9e'
    'Xd89Su+1Qu3eUHrQOagHHCGPaq9aFJ0P7WFbBiYb8gYdaRblITZv/2Q=="},"rows":36}'))

HOOK_SOURCE = r'''"""knee-hb126 codec hook v1.0.0 (module knee_hb126_hook; H-B-126, tools/hb126_codecs.py).

Imported at interpreter start by the usercustomize/sitecustomize entry next to it (its directory is
on PYTHONPATH); importing it twice is a no-op (module cache). Appends the private python-gdcm
directory to sys.path and arms a one-shot finder: after pydicom.pixels.decoders.base executes, the
'gdcm' plugin goes LAST wherever another plugin was available (whatever decoded before keeps its
plugin) and every compressed plugin is wrapped to count successes/failures per KNEE_CODEC_BRANCH
into KNEE_CODEC_TELEMETRY. Every step is guarded; an error leaves pydicom unchanged.
"""
import os as _hb126_os
import sys as _hb126_sys

__HB126_SITE__ = None
KNEE_HB126_STATE = {"version": "1.0.0", "site": __HB126_SITE__, "armed": False, "patched": False,
                    "demoted": [], "orders": {}, "errors": [], "counts": {}, "final_failures": {},
                    "error_samples": [], "entries": [], "chained": []}


def chain(name, entry_file):
    """Run the next `name` module on sys.path (skipping the entry's own directory) as `name`."""
    import importlib.machinery
    import importlib.util
    here = _hb126_os.path.normcase(_hb126_os.path.dirname(_hb126_os.path.abspath(entry_file)))
    KNEE_HB126_STATE["entries"].append(name)
    paths = [p for p in _hb126_sys.path
             if _hb126_os.path.normcase(_hb126_os.path.abspath(p or _hb126_os.getcwd())) != here]
    spec = importlib.machinery.PathFinder.find_spec(name, paths)
    if spec is None or spec.loader is None:
        return None
    ours = _hb126_sys.modules.get(name)
    module = importlib.util.module_from_spec(spec)
    _hb126_sys.modules[name] = module  # the import system re-reads sys.modules after our entry ran
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if ours is not None:
            _hb126_sys.modules[name] = ours
        raise
    KNEE_HB126_STATE["chained"].append(str(spec.origin))
    return spec.origin


def _hb126_install():
    import importlib.abc
    import threading
    import time

    state = KNEE_HB126_STATE
    target_name = "pydicom.pixels.decoders.base"
    io = {"lock": threading.RLock(), "last": 0.0, "pid": _hb126_os.getpid(),
          "token": _hb126_os.urandom(4).hex()}

    def note(message):
        if len(state["errors"]) < 20:
            state["errors"].append(str(message)[:300])

    def reset():
        io.update(lock=threading.RLock(), last=0.0, pid=_hb126_os.getpid(), token=_hb126_os.urandom(4).hex())
        state.update(counts={}, final_failures={}, error_samples=[])

    def flush(force=False):
        directory = _hb126_os.environ.get("KNEE_CODEC_TELEMETRY")
        if not directory:
            return
        now = time.monotonic()
        if not force and now - io["last"] < 2.0:
            return
        io["last"] = now
        try:
            import json
            pyd = _hb126_sys.modules.get("pydicom")
            gmod = _hb126_sys.modules.get("pydicom.pixels.decoders.gdcm")
            with io["lock"]:
                record = json.loads(json.dumps(state, default=str))
            record.update(pid=_hb126_os.getpid(), ppid=_hb126_os.getppid(), updated=time.time(),
                          argv=[str(a)[:80] for a in list(getattr(_hb126_sys, "argv", []) or [])[:3]],
                          executable=_hb126_sys.executable, branch=_hb126_os.environ.get("KNEE_CODEC_BRANCH"),
                          pydicom=getattr(pyd, "__version__", None),
                          gdcm=(".".join(map(str, getattr(gmod, "GDCM_VERSION", ())))
                                if getattr(gmod, "HAVE_GDCM", False) else None))
            _hb126_os.makedirs(directory, exist_ok=True)
            path = _hb126_os.path.join(directory, "codec-%d-%s.json" % (record["pid"], io["token"]))
            with open(path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True)
            _hb126_os.replace(path + ".tmp", path)
        except Exception as exc:
            note("flush: %r" % (exc,))

    def event(uid, label, ok, last, exc=None):
        try:
            branch = _hb126_os.environ.get("KNEE_CODEC_BRANCH") or "unlabelled"
            if io["pid"] != _hb126_os.getpid():  # forked without the at-fork reset
                reset()
            with io["lock"]:
                per = state["counts"].setdefault(branch, {}).setdefault(uid, {}).setdefault(label, [0, 0])
                per[0 if ok else 1] += 1
                if not ok and last:
                    fin = state["final_failures"].setdefault(branch, {})
                    fin[uid] = fin.get(uid, 0) + 1
                if not ok and len(state["error_samples"]) < 8:
                    state["error_samples"].append(("%s %s %s %s: %s" % (branch, uid, label, type(exc).__name__, exc))[:300])
                flush(force=(not ok) or sum(per) == 1)
        except Exception as error:
            note("event: %r" % (error,))

    def counted(uid, label, fn, last):
        def plugin(src, runner):
            try:
                result = fn(src, runner)
            except BaseException as exc:
                event(uid, label, False, last, exc)
                raise
            event(uid, label, True, last)
            return result
        plugin.__wrapped__ = fn
        plugin.__name__ = getattr(fn, "__name__", "plugin")
        return plugin

    def patch(base):
        decoder_type = base.Decoder
        seen = set()
        members = vars(base)
        for name in sorted(members):
            obj = members[name]
            if not isinstance(obj, decoder_type) or id(obj) in seen:
                continue
            seen.add(id(obj))
            available = getattr(obj, "_available", None)
            if not isinstance(available, dict) or not available:
                continue
            uid = str(obj.UID)
            labels = list(available)
            if "gdcm" in available and len(labels) > 1 and labels[-1] != "gdcm":
                available["gdcm"] = available.pop("gdcm")  # fallback only where another plugin exists
                state["demoted"].append(uid)
            labels = list(available)
            for label in labels:
                available[label] = counted(uid, label, available[label], label == labels[-1])
            state["orders"][uid] = labels
        state["patched"] = True
        if hasattr(_hb126_os, "register_at_fork"):
            _hb126_os.register_at_fork(after_in_child=reset)
        import atexit
        atexit.register(flush, True)
        flush(True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != target_name or not state["armed"]:
                return None
            state["armed"] = False
            try:
                spec = None
                for finder in list(_hb126_sys.meta_path):
                    find = getattr(finder, "find_spec", None)
                    if finder is self or find is None:
                        continue
                    spec = find(fullname, path, target)
                    if spec is not None:
                        break
                loader = getattr(spec, "loader", None)
                inner = getattr(loader, "exec_module", None)
                if inner is None:
                    note("no loader for " + fullname)
                    return None
                finder_self = self

                def exec_module(module):
                    inner(module)
                    try:
                        patch(module)
                    except Exception as exc:
                        note("patch: %r" % (exc,))
                    try:
                        _hb126_sys.meta_path.remove(finder_self)
                    except ValueError:
                        pass
                loader.exec_module = exec_module
                return spec
            except Exception as exc:
                note("find_spec: %r" % (exc,))
                return None

    site = state["site"]
    if site and _hb126_os.path.isdir(site) and site not in _hb126_sys.path:
        _hb126_sys.path.append(site)
    if target_name in _hb126_sys.modules:  # imported before the hook (should not happen)
        patch(_hb126_sys.modules[target_name])
    else:
        state["armed"] = True
        _hb126_sys.meta_path.insert(0, _Finder())


try:
    _hb126_install()
except Exception as _hb126_exc:  # never break interpreter start-up
    KNEE_HB126_STATE["errors"].append("install: %r" % (_hb126_exc,))
'''

ENTRY_SOURCE = r'''"""knee-hb126 codec hook entry v1.0.0 (__NAME__; H-B-126, tools/hb126_codecs.py).

Imports knee_hb126_hook (same directory), then runs the next __NAME__ module on sys.path.
"""
try:
    import knee_hb126_hook as _hb126_hook
except Exception as _hb126_exc:
    import sys as _hb126_sys
    _hb126_sys.stderr.write("knee-hb126 hook import failed: %r" % (_hb126_exc,) + chr(10))
else:
    _hb126_hook.chain("__NAME__", __file__)
'''

VERIFY_SOURCE = r'''
import base64, hashlib, json, os, sys
out = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import site as _site
    out["user_site_enabled"] = bool(getattr(_site, "ENABLE_USER_SITE", False))
    uc = sys.modules.get("knee_hb126_hook")
    out["hook_file"] = getattr(uc, "__file__", None)
    st = getattr(uc, "KNEE_HB126_STATE", None)
    out["customize_modules"] = {n: getattr(sys.modules.get(n), "__file__", None) for n in ("sitecustomize", "usercustomize")}
    import numpy as np
    import pydicom
    from pydicom.pixels.decoders import base as _base
    out["pydicom"] = pydicom.__version__
    out["pydicom_file"] = pydicom.__file__
    try:
        import gdcm
        out["gdcm"] = gdcm.Version.GetVersion()
        out["gdcm_file"] = gdcm.__file__
    except Exception as exc:
        out["gdcm_error"] = (type(exc).__name__ + ": " + str(exc))[:300]
    out["hook"] = None if st is None else {k: st.get(k) for k in ("version", "site", "patched", "demoted", "errors", "entries", "chained")}
    orders, seen = {}, set()
    for name, obj in sorted(vars(_base).items()):
        if isinstance(obj, _base.Decoder) and id(obj) not in seen:
            seen.add(id(obj))
            if obj._available:
                orders[str(obj.UID)] = list(obj._available)
    out["orders"] = orders
    spec = json.loads(sys.argv[1])
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import encapsulate
    results = {}
    for uid, b64 in sorted(spec["frames"].items()):
        rec = {}
        try:
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = uid
            ds.Rows, ds.Columns = spec["rows"], spec["columns"]
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, spec["bits_stored"], spec["bits_stored"] - 1
            ds.PixelRepresentation = 0
            ds.PixelData = encapsulate([base64.b64decode(b64)])
            ds["PixelData"].VR = "OB"
            ds["PixelData"].is_undefined_length = True
            arr = np.asarray(ds.pixel_array)
            rec["shape"] = list(arr.shape)
            rec["sha256"] = hashlib.sha256(np.ascontiguousarray(arr.astype("<u2")).tobytes()).hexdigest()
            rec["equal"] = rec["shape"] == [spec["rows"], spec["columns"]] and rec["sha256"] == spec["array_sha256"]
        except Exception as exc:
            rec["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
            rec["equal"] = False
        results[uid] = rec
    out["selftest"] = results
    if st is not None:
        out["selftest_counts"] = st.get("counts")
except Exception as exc:
    out["error"] = (type(exc).__name__ + ": " + str(exc))[:500]
print("HB126_CODEC_VERIFY " + json.dumps(out, sort_keys=True, default=str), flush=True)
'''


# ---------------------------------------------------------------------------------------- helpers
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_source(name) -> str:
    if name not in ENTRY_NAMES:
        raise ValueError(f"unknown entry {name!r}")
    return ENTRY_SOURCE.replace("__NAME__", name)


def hook_source(site) -> str:
    """The knee_hb126_hook module text bound to one private gdcm directory."""
    anchor = "__HB126_SITE__ = None\n"
    if HOOK_SOURCE.count(anchor) != 1:
        raise ValueError("hook anchor changed")
    return HOOK_SOURCE.replace(anchor, f"__HB126_SITE__ = {str(site)!r}\n")


def safe_member(name: str) -> bool:
    if not name or name.startswith(("/", "\\")) or "\\" in name or ":" in name or "\x00" in name:
        return False
    parts = name.split("/")
    return all(p not in ("", ".", "..") for p in (parts[:-1] if name.endswith("/") else parts))


def safe_unzip(archive, dest) -> list[str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    root = dest.resolve()
    names = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if not safe_member(info.filename) or (target != root and root not in target.parents):
                raise ValueError(f"unsafe archive member: {info.filename!r}")
        for info in z.infolist():
            z.extract(info, dest)
            if not info.is_dir():
                names.append(info.filename)
    return names


def find_wheel(input_root="/kaggle/input", wheel=WHEEL, max_depth: int = 6):
    """First file named exactly like the pinned wheel below input_root (DICOM trees pruned)."""
    root = Path(input_root)
    if not root.is_dir():
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if d not in SKIP_WALK)
        if wheel["file"] in filenames:
            return Path(dirpath) / wheel["file"]
    return None


def child_env(base_env, receipt, *, branch, telemetry_dir=None) -> dict:
    """base_env plus the hook directory (appended LAST to PYTHONPATH) when the receipt is ACTIVE.

    Any other status returns an unchanged copy, i.e. the pre-codec environment.
    """
    env = dict(base_env)
    if not receipt or receipt.get("status") != "ACTIVE":
        return env
    entries = [e for e in env.get("PYTHONPATH", "").split(os.pathsep) if e]
    if receipt["hook_dir"] not in entries:
        entries.append(receipt["hook_dir"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env[BRANCH_ENV] = str(branch)
    if telemetry_dir:
        env[TELEMETRY_ENV] = str(telemetry_dir)
    return env


def strip_env(env, receipt) -> dict:
    """Undo child_env: the hook directory and the codec variables are removed."""
    env = dict(env)
    hook = (receipt or {}).get("hook_dir")
    if "PYTHONPATH" in env:
        kept = [e for e in env["PYTHONPATH"].split(os.pathsep) if e and e != hook]
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            env.pop("PYTHONPATH")
    env.pop(BRANCH_ENV, None)
    env.pop(TELEMETRY_ENV, None)
    return env


def verify(receipt, *, python=None, base_env=None, branch="verify", telemetry_dir=None) -> dict:
    """Fresh interpreter with the hook: decoder orders, gdcm import and the embedded self-test."""
    env = child_env(dict(os.environ if base_env is None else base_env), dict(receipt, status="ACTIVE"),
                    branch=branch, telemetry_dir=telemetry_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    try:
        proc = subprocess.run([python or sys.executable, "-c", VERIFY_SOURCE, json.dumps(SELFTEST)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300], "seconds": round(time.monotonic() - started, 2)}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("HB126_CODEC_VERIFY ")]
    result = json.loads(lines[-1].split(" ", 1)[1]) if lines else {"error": "no verify line"}
    result.update(returncode=proc.returncode, seconds=round(time.monotonic() - started, 2))
    if proc.returncode or not lines:
        result["output_tail"] = proc.stdout[-1500:]
    return result


def evaluate(result, receipt) -> tuple[str, list[str]]:
    """ACTIVE only when the fresh interpreter shows the complete intended configuration."""
    reasons = []
    hook_file = result.get("hook_file")
    expected = os.path.normcase(os.path.abspath(Path(receipt["hook_dir"]) / (HOOK_MODULE + ".py")))
    if result.get("error"):
        reasons.append("VERIFY_ERROR")
    if not hook_file or os.path.normcase(os.path.abspath(hook_file)) != expected:
        reasons.append("HOOK_NOT_LOADED")
    hook = result.get("hook") or {}
    if not hook.get("patched") or hook.get("errors"):
        reasons.append("HOOK_NOT_PATCHED")
    if not result.get("gdcm"):
        reasons.append("GDCM_IMPORT")
    orders = result.get("orders") or {}
    if any("gdcm" not in orders.get(uid, []) for uid in TARGET_UIDS):
        reasons.append("GDCM_NOT_REGISTERED")
    if any("gdcm" in labels and len(labels) > 1 and labels[-1] != "gdcm" for labels in orders.values()):
        reasons.append("GDCM_NOT_LAST")
    tests = result.get("selftest") or {}
    if set(tests) != set(SELFTEST["frames"]) or not all(t.get("equal") for t in tests.values()):
        reasons.append("SELFTEST")
    return ("ACTIVE" if not reasons else "INACTIVE_" + reasons[0]), reasons


def install(wheel_path, root, *, wheel=WHEEL, method="auto", python=None, base_env=None,
            telemetry_dir=None) -> dict:
    """Never raises. Fresh root -> root/site (gdcm) + root/hook (hook module + one entry), then verify().

    Entry preference: usercustomize.py; if the fresh interpreter did not load the hook through it
    (e.g. user site disabled), the entry is replaced by sitecustomize.py and verified again.
    """
    started = time.monotonic()
    rec = {"version": VERSION, "hook_version": HOOK_VERSION, "status": "INACTIVE_ERROR", "root": str(root),
           "wheel": {"expected_file": wheel["file"], "expected_sha256": wheel["sha256"],
                     "path": str(wheel_path) if wheel_path else None},
           "method": None, "attempts": []}
    try:
        if wheel_path is None or not Path(wheel_path).is_file():
            rec["status"] = "INACTIVE_NO_WHEEL"
            return rec
        wheel_path = Path(wheel_path).absolute()
        rec["wheel"].update(path=str(wheel_path), sha256=sha256_file(wheel_path), size=wheel_path.stat().st_size)
        if (wheel_path.name != wheel["file"] or rec["wheel"]["sha256"] != wheel["sha256"]
                or rec["wheel"]["size"] != wheel["size"]):
            rec["status"] = "INACTIVE_WHEEL_PIN"
            return rec
        root = Path(root).absolute()  # the hook embeds this path; children may run elsewhere
        rec["root"] = str(root)
        root.mkdir(parents=True, exist_ok=False)
        site, hook = root / "site", root / "hook"
        for m in (["pip", "unzip"] if method == "auto" else [method]):
            if site.exists():
                shutil.rmtree(site)
            attempt = {"method": m}
            try:
                if m == "pip":
                    env = dict(os.environ if base_env is None else base_env, PYTHONDONTWRITEBYTECODE="1")
                    cmd = [python or sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                           "--no-compile", "--disable-pip-version-check", "--target", str(site), str(wheel_path)]
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                    attempt.update(returncode=proc.returncode, log_tail=proc.stdout[-1200:])
                elif m == "unzip":
                    attempt["files"] = len(safe_unzip(wheel_path, site))
                else:
                    raise ValueError(f"unknown method {m!r}")
                attempt["ok"] = (site / "gdcm.py").is_file() and (site / "_gdcm").is_dir() and not attempt.get("returncode")
            except Exception as exc:
                attempt.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
            rec["attempts"].append(attempt)
            if attempt["ok"]:
                rec["method"] = m
                break
        if rec["method"] is None:
            rec["status"] = "INACTIVE_INSTALL"
            return rec
        hook.mkdir()
        module = hook / (HOOK_MODULE + ".py")
        module.write_text(hook_source(site), encoding="utf-8")
        rec.update(site=str(site), hook_dir=str(hook), hook_sha256=sha256_file(module), entries_tried=[])
        for name in ENTRY_NAMES:
            for other in ENTRY_NAMES:
                (hook / (other + ".py")).unlink(missing_ok=True)
            (hook / (name + ".py")).write_text(entry_source(name), encoding="utf-8")
            result = verify(rec, python=python, base_env=base_env, telemetry_dir=telemetry_dir)
            status, reasons = evaluate(result, rec)
            rec["entries_tried"].append({"entry": name, "status": status, "reasons": reasons})
            rec.update(entry=name, verify=result, status=status, reasons=reasons)
            if status == "ACTIVE" or "HOOK_NOT_LOADED" not in reasons:
                break
        if rec["status"] != "ACTIVE":  # leave no entry behind: the hook dir is then inert
            for other in ENTRY_NAMES:
                (hook / (other + ".py")).unlink(missing_ok=True)
            rec["entry"] = None
        else:
            rec["entry_sha256"] = sha256_file(hook / (rec["entry"] + ".py"))
    except Exception as exc:
        rec.update(status="INACTIVE_ERROR", error=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        rec["seconds"] = round(time.monotonic() - started, 2)
    return rec


def read_telemetry(directory) -> dict:
    """Aggregate the hook's per-process files: per-branch decoded frames and failures."""
    out = {"present": Path(directory).is_dir(), "files": 0, "unreadable": [], "processes": [], "branches": {},
           "error_samples": [], "hook_errors": [], "orders_variants": {},
           "totals": {"decoded_compressed_frames": 0, "plugin_failures": 0, "decode_failures": 0}}
    if not out["present"]:
        return out
    for path in sorted(Path(directory).glob("codec-*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out["unreadable"].append(path.name)
            continue
        out["files"] += 1
        orders = json.dumps(rec.get("orders", {}), sort_keys=True)
        digest = hashlib.sha256(orders.encode()).hexdigest()[:16]
        out["orders_variants"].setdefault(digest, json.loads(orders))
        out["processes"].append({k: rec.get(k) for k in ("pid", "ppid", "argv", "branch", "pydicom", "gdcm",
                                                          "patched", "demoted")} | {"orders": digest})
        out["hook_errors"] += [f"{rec.get('pid')}: {e}" for e in rec.get("errors", [])][:5]
        out["error_samples"] += rec.get("error_samples", [])[:5]
        for branch, per_uid in rec.get("counts", {}).items():
            agg = out["branches"].setdefault(branch, {"decoded_compressed_frames": 0, "plugin_failures": 0,
                                                      "decode_failures": 0, "by_ts": {}})
            for uid, per_label in per_uid.items():
                for label, (ok, fail) in per_label.items():
                    agg["decoded_compressed_frames"] += int(ok)
                    agg["plugin_failures"] += int(fail)
                    cell = agg["by_ts"].setdefault(uid, {}).setdefault(label, [0, 0])
                    cell[0] += int(ok)
                    cell[1] += int(fail)
        for branch, per_uid in rec.get("final_failures", {}).items():
            agg = out["branches"].setdefault(branch, {"decoded_compressed_frames": 0, "plugin_failures": 0,
                                                      "decode_failures": 0, "by_ts": {}})
            agg["decode_failures"] += sum(int(v) for v in per_uid.values())
    out["error_samples"] = out["error_samples"][:20]
    out["hook_errors"] = out["hook_errors"][:20]
    out["totals"] = {k: sum(b[k] for b in out["branches"].values())
                     for k in ("decoded_compressed_frames", "plugin_failures", "decode_failures")}
    return out


def default_root(prefix="knee-hb126-codecs") -> Path:
    """A fresh private directory outside /kaggle/working (never saved as kernel output)."""
    import tempfile
    for base in ("/kaggle/tmp", "/tmp"):
        if os.path.isdir(base) and os.access(base, os.W_OK):
            return Path(tempfile.mkdtemp(prefix=prefix + "-", dir=base)) / "root"
    return Path(tempfile.mkdtemp(prefix=prefix + "-")) / "root"


def activate(input_root="/kaggle/input", *, root=None, telemetry_dir=None, base_env=None) -> dict:
    """find_wheel + install; never raises (the release parent's entry point)."""
    try:
        wheel = find_wheel(input_root)
        return install(wheel, root or default_root(), base_env=base_env, telemetry_dir=telemetry_dir)
    except Exception as exc:
        return {"version": VERSION, "status": "INACTIVE_ERROR", "error": f"{type(exc).__name__}: {exc}"[:300]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=f"HB126 codecs v{VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("install")
    a.add_argument("--wheel", required=True)
    a.add_argument("--root", required=True)
    a.add_argument("--method", default="auto", choices=["auto", "pip", "unzip"])
    a.add_argument("--telemetry")
    t = sub.add_parser("telemetry")
    t.add_argument("directory")
    args = ap.parse_args(argv)
    if args.cmd == "install":
        print(json.dumps(install(args.wheel, args.root, method=args.method, telemetry_dir=args.telemetry), indent=1))
    else:
        print(json.dumps(read_telemetry(args.directory), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
