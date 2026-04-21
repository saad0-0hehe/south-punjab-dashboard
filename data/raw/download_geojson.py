import requests, json, zipfile, io, os

url = 'https://data.humdata.org/api/3/action/package_show?id=cod-ab-pak'
r = requests.get(url, timeout=15)
pkg = r.json()['result']

for res in pkg.get('resources', []):
    name = res.get('name', '')
    if 'geojson' in name.lower():
        dl_url = res['url']
        print(f"Downloading: {name}")
        print(f"URL: {dl_url}")
        
        r2 = requests.get(dl_url, timeout=60, stream=True)
        print(f"Status: {r2.status_code}, Size: {len(r2.content)} bytes")
        
        if r2.status_code == 200:
            # It's a zip, extract it
            if name.endswith('.zip') or 'zip' in r2.headers.get('Content-Type',''):
                z = zipfile.ZipFile(io.BytesIO(r2.content))
                for fname in z.namelist():
                    print(f"  Zip contains: {fname}")
                    if 'adm2' in fname.lower() and fname.endswith('.json'):
                        z.extract(fname, 'data/')
                        print(f"  Extracted: data/{fname}")
            else:
                with open('data/pak_adm2.geojson', 'wb') as f:
                    f.write(r2.content)
                print("Saved directly")
        break
