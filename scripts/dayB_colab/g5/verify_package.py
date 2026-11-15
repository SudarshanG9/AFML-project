from common import verify_code

if __name__ == '__main__':
    manifest = verify_code()
    print('PACKAGE INTEGRITY PASS:', len(manifest['files']), 'files; no G5 work executed')
