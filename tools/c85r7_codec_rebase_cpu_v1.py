"""v1.0.0: real DICOM/real-checkpoint CPU flip smoke on the codec-only base."""
import json
from pathlib import Path


def main():
    # Reuse the reviewed native comparison without its old deadline-base setup.
    root = Path(__file__).resolve().parents[1]
    source = (root/'tools/exp169_c85r7flip_cpu_v130.py').read_text(encoding='utf-8')
    source = source.replace('exp169_release_c85r7flip_v130 as build', 'c85r7_codec_rebase_v1 as build')
    source = source.replace('exp169_release_c85r7codec_deadline_v122 as base', 'exp169_release_c85r7codec as base')
    source = source.replace('build.read(', 'build.h.read(').replace('build.encoded(', 'encoded(')
    source = source.replace('for variant in flip.VARIANTS:', "for variant in ['FLIP_ALL']:")
    source = source.replace("runtime = runtimes['FLIP_F0F2']", "runtime = runtimes['FLIP_ALL']")
    source = source.replace('with base.metadata_io.writers(base):', 'if True:')
    source = source.replace('flip.installed(base.r7, variant, base.metadata_io)', 'flip.installed(base.r7, variant, build.metadata_io)')
    source = source.replace('[True, False, True, False]', '[True, True, True, True]')
    scope = dict(__name__='codec_rebase_cpu_smoke', __file__=__file__,
                 encoded=lambda value: (json.dumps(value, indent=2)+'\n').encode())
    exec(compile(source, __file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
