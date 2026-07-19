"""Phase G multi-device encrypted sync — optional ``[sync]`` extra.

This subpackage is NOT part of the stdlib floor tier. Every module here MUST
be importable without the ``cryptography`` dependency present; the actual
crypto primitives are imported lazily (inside functions/methods) so that a
mere ``import brain.sync.crypto`` never pulls in ``cryptography``. A clear
error is raised only when a crypto operation is genuinely attempted without
the dependency installed. Install the extra with ``pip install -e .[sync]``.
"""
