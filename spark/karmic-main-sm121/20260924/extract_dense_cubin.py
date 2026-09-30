"""Extract the single embedded CUDA ELF from a retained CuTe host object."""
import argparse
from pathlib import Path
import struct

p = argparse.ArgumentParser()
p.add_argument('object')
p.add_argument('output')
a = p.parse_args()
data = Path(a.object).read_bytes()
target = Path(a.output)
if target.exists():
    raise RuntimeError('Refuse to overwrite retained cubin')
found = []
cursor = 0
while True:
    cursor = data.find(b'\x7fELF', cursor)
    if cursor < 0:
        break
    if data[cursor + 4:cursor + 6] == b'\x02\x01' and struct.unpack_from('<H', data, cursor + 18)[0] == 190:
        fields = struct.unpack_from('<16sHHIQQQIHHHHHH', data, cursor)
        phoff, shoff = fields[5], fields[6]
        phentsize, phnum, shentsize, shnum = fields[9:13]
        end = max(64, phoff + phentsize * phnum, shoff + shentsize * shnum)
        for i in range(shnum):
            section = struct.unpack_from('<IIQQQQIIQQ', data, cursor + shoff + i * shentsize)
            if section[1] != 8:  # SHT_NOBITS has no file payload.
                end = max(end, section[4] + section[5])
        if cursor + end > len(data):
            raise RuntimeError('Truncated embedded ELF')
        found.append(data[cursor:cursor + end])
    cursor += 4
if len(found) != 1:
    raise RuntimeError(f'Expected one CUDA object, found {len(found)}')
target.write_bytes(found[0])
print(target, len(found[0]))
