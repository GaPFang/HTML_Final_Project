import requests
import json
from datetime import timedelta, date
from tqdm import tqdm

api_key = "a6cc4ea08a5746e8bf361044232711"
lat = "25.0173"
lon = "121.5398"

start_date = date(2023, 10, 1)  
end_date = date(2023, 11, 26)

delta = end_date - start_date

for i in tqdm(range(delta.days + 1)):
    day = start_date + timedelta(days=i)
    url_date = day.strftime("%Y-%m-%d")
    
    url = f"https://api.weatherapi.com/v1/history.json?key={api_key}&q={lat},{lon}&dt={url_date}"
    response = requests.get(url)
    data = json.loads(response.text)
    
    date = day.strftime("%Y%m%d")
    with open(f"history_{date}.json", "w") as f:
        json.dump(data, f)

    # print(f"Downloaded weather data for {url_date}")