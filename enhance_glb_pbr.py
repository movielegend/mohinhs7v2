"""Add clear-coat and specular material extensions to an exported binary glTF."""
import json, struct, sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
magic, version, total = struct.unpack_from('<4sII', data, 0)
assert magic == b'glTF' and version == 2 and total == len(data)
json_len, json_type = struct.unpack_from('<II', data, 12)
assert json_type == 0x4E4F534A
start = 20
doc = json.loads(data[start:start + json_len].decode('utf-8').rstrip(' \0'))
tail = data[start + json_len:]

used = doc.setdefault('extensionsUsed', [])
for extension in ('KHR_materials_clearcoat', 'KHR_materials_specular'):
    if extension not in used:
        used.append(extension)

for mat in doc.get('materials', []):
    name = mat.get('name', '').lower()
    ext = mat.setdefault('extensions', {})
    if 'glass' in name or 'optical' in name:
        clear, clear_rough, spec = .95, .035, 1.0
    elif 'charcoal' in name or 'lens anodized' in name:
        clear, clear_rough, spec = .62, .10, .75
    elif 'housing' in name:
        clear, clear_rough, spec = .24, .32, .42
    elif 'support' in name:
        clear, clear_rough, spec = .18, .40, .36
    elif 'aluminium' in name or 'silver' in name:
        clear, clear_rough, spec = .15, .20, .85
    else:
        clear, clear_rough, spec = .08, .45, .50
    ext['KHR_materials_clearcoat'] = {'clearcoatFactor': clear, 'clearcoatRoughnessFactor': clear_rough}
    ext['KHR_materials_specular'] = {'specularFactor': spec}

blob = json.dumps(doc, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
blob += b' ' * ((4 - len(blob) % 4) % 4)
out = bytearray(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(blob) + len(tail)))
out += struct.pack('<II', len(blob), 0x4E4F534A)
out += blob
out += tail
path.write_bytes(out)
print(f'Enhanced {path.name}: {len(doc.get("materials", []))} materials')
