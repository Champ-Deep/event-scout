"""Read-only dump of the Railway Postgres to stdout (run inside the Railway service; see import_railway.py)."""
import os,asyncio,asyncpg,json,base64,sys,datetime,uuid,decimal
def conv(v):
    if isinstance(v,(bytes,memoryview)): return {"__b64__":base64.b64encode(bytes(v)).decode()}
    if isinstance(v,datetime.datetime):
        if v.tzinfo is None: v=v.replace(tzinfo=datetime.timezone.utc)
        return v.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.')+f"{v.microsecond//1000:03d}Z"
    if isinstance(v,uuid.UUID): return str(v)
    if isinstance(v,decimal.Decimal): return float(v)
    return v
async def m():
    c=await asyncpg.connect(os.environ['DATABASE_URL'])
    out={"schema":{}, "data":{}}
    for r in await c.fetch("select table_name,column_name,data_type,is_nullable,column_default from information_schema.columns where table_schema='public' order by table_name,ordinal_position"):
        out["schema"].setdefault(r['table_name'],[]).append([r['column_name'],r['data_type'],r['is_nullable'],r['column_default']])
    for t in out["schema"]:
        rows=await c.fetch(f'select * from "{t}"')
        out["data"][t]=[{k:(json.loads(v) if isinstance(v,str) and k in () else conv(v)) for k,v in dict(r).items()} for r in rows]
    sys.stdout.write("@@BEGIN@@"+json.dumps(out)+"@@END@@")
asyncio.run(m())
