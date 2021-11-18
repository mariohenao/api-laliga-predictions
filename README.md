# LaLiga match predictor API

This container launches an API that predicts the outcome of a given match of 2 LaLiga clubs and the match date.

## Docker

### Build the container

"api-laliga" is the name of the container (you can use anohter name)
- `docker build -t api-laliga .`

### Run the container

- `sudo docker run -dp 8080:8080 api-laliga:latest`

## Try the API
In your Browser go to `localhost:8080/docs` and try the POST method "/predict"

### Example data
input: `{"match": {"local": "Valencia", "visitor": "Real Mdrid", "date": "2004/12/06"}}`
- local: team playing as a local
- visitor: team playin as a visitor
- date: match date

output: `{"winner": "local","confidence": "0.4601581"}`
- winner: local - local team wins; visitor - visitor team wins; draw - teams drawn

You can use othe values for "local", "visitor" and "date". 

In this version, the team names are case sensitive, here some names to try:
- Barcelona
- Real Madrid
- Atletico de Madid
- Valencia
- Athletic Club
- Real Sociedad
- Sevilla
- Deportivo
- Villareal
- Lleida

Please enter the date with format `YYYY/mm/dd`.
