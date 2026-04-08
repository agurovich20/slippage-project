from fetch_trades import API_KEY
from polygon import RESTClient

client = RESTClient(api_key=API_KEY)
exchanges = client.get_exchanges(asset_class="stocks")

print(f"{'ID':>4}  {'MIC':<12}  {'Name':<40}  {'Type'}")
print("-" * 75)
for ex in sorted(exchanges, key=lambda e: e.id or 0):
    print(f"{ex.id or '?':>4}  {ex.mic or ''::<12}  {ex.name or ''::<40}  {ex.type or ''}")
