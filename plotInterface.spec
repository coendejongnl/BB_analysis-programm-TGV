# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['plotInterface.py'],
             pathex=['C:\\Users\\tumuz\\git\\BlueBatteryPlotter'],
             binaries=[],
             datas=[],
             hiddenimports=['PyQt5', 'sklearn', 'sklearn.utils._cython_blas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='plotInterface',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , icon='batteryIcon.ico')
