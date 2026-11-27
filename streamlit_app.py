import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Прогноз выборов США — 2028", page_icon="🗳️", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ELECTION_YEAR = 2028

STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

# 2020 Census apportionment, in effect for the 2024 and 2028 elections.
ELECTORAL_VOTES = {
    "AL": 9,
    "AK": 3,
    "AZ": 11,
    "AR": 6,
    "CA": 54,
    "CO": 10,
    "CT": 7,
    "DE": 3,
    "DC": 3,
    "FL": 30,
    "GA": 16,
    "HI": 4,
    "ID": 4,
    "IL": 19,
    "IN": 11,
    "IA": 6,
    "KS": 6,
    "KY": 8,
    "LA": 8,
    "ME": 4,
    "MD": 10,
    "MA": 11,
    "MI": 15,
    "MN": 10,
    "MS": 6,
    "MO": 10,
    "MT": 4,
    "NE": 5,
    "NV": 6,
    "NH": 4,
    "NJ": 14,
    "NM": 5,
    "NY": 28,
    "NC": 16,
    "ND": 3,
    "OH": 17,
    "OK": 7,
    "OR": 8,
    "PA": 19,
    "RI": 4,
    "SC": 9,
    "SD": 3,
    "TN": 11,
    "TX": 40,
    "UT": 6,
    "VT": 3,
    "VA": 13,
    "WA": 12,
    "WV": 4,
    "WI": 10,
    "WY": 3,
}


def fetch_state_forecast(api_url: str, state: str, year: int) -> dict:
    response = requests.get(
        f"{api_url}/predict/{state}",
        params={"year": year},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner="Загружаем прогнозы по штатам...")
def load_forecasts(api_url: str, year: int) -> list[dict]:
    states = sorted(STATE_NAMES)
    with ThreadPoolExecutor(max_workers=12) as executor:
        return list(executor.map(lambda state: fetch_state_forecast(api_url, state, year), states))


def electoral_vote_totals(forecasts: list[dict]) -> dict[str, int]:
    totals = {"Democrat": 0, "Republican": 0}
    for forecast in forecasts:
        totals[forecast["winner"]] += ELECTORAL_VOTES[forecast["state"]]
    return totals


def render_forecast_map(forecasts: list[dict]) -> None:
    data = pd.DataFrame(forecasts)
    data["state_name"] = data["state"].map(STATE_NAMES)
    data["electoral_votes"] = data["state"].map(ELECTORAL_VOTES)

    figure = px.choropleth(
        data,
        locations="state",
        locationmode="USA-states",
        color="winner",
        color_discrete_map={"Democrat": "#2563eb", "Republican": "#dc2626"},
        scope="usa",
        hover_name="state_name",
        hover_data={
            "state": True,
            "winner": True,
            "probability_democrat": ":.1%",
            "probability_republican": ":.1%",
            "electoral_votes": True,
        },
        labels={
            "state": "Код",
            "winner": "Победитель",
            "probability_democrat": "Вероятность демократов",
            "probability_republican": "Вероятность республиканцев",
            "electoral_votes": "Выборщики",
        },
    )
    figure.update_traces(marker_line_color="white", marker_line_width=0.8)
    figure.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        legend_title_text="Прогноз",
        geo={"showlakes": True, "lakecolor": "white"},
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_custom_prediction() -> None:
    st.subheader("Свой сценарий")
    st.caption("Измените признаки и отправьте их в FastAPI для отдельного прогноза.")

    with st.form("custom_prediction"):
        incumbent_party = st.selectbox(
            "Партия действующего президента",
            options=["R", "D"],
            format_func=lambda party: "Республиканцы" if party == "R" else "Демократы",
        )
        col1, col2 = st.columns(2)
        with col1:
            previous_share = st.slider(
                "Доля партии власти в штате на прошлых выборах",
                min_value=0.0,
                max_value=1.0,
                value=0.509,
                step=0.001,
                format="%.3f",
            )
            previous_lean = st.slider(
                "Преимущество штата перед страной",
                min_value=-1.0,
                max_value=1.0,
                value=0.001,
                step=0.001,
                format="%.3f",
            )
            home_state = st.selectbox(
                "Родной штат кандидата",
                options=[0, 1, -1],
                format_func={
                    0: "Нет",
                    1: "Кандидат партии власти",
                    -1: "Кандидат оппозиции",
                }.get,
            )
        with col2:
            incumbent_running = st.selectbox("Президент баллотируется повторно", [0, 1], format_func=lambda value: "Нет" if value == 0 else "Да")
            long_tenure = st.selectbox("Партия у власти два срока или дольше", [0, 1], format_func=lambda value: "Нет" if value == 0 else "Да")

        submitted = st.form_submit_button("Рассчитать прогноз", type="primary")

    if not submitted:
        return

    payload = {
        "incumbent_party": incumbent_party,
        "inc_prev_share": previous_share,
        "inc_prev_lean": previous_lean,
        "inc_running": incumbent_running,
        "inc_long_tenure": long_tenure,
        "inc_home_state": home_state,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()
    except requests.RequestException as error:
        st.error(f"Не удалось получить прогноз от API: {error}")
        return

    winner = "Демократы" if prediction["winner"] == "Democrat" else "Республиканцы"
    st.success(f"Прогноз: {winner}")
    democrat_col, republican_col = st.columns(2)
    democrat_col.metric("Вероятность демократов", f"{prediction['probability_democrat']:.1%}")
    republican_col.metric("Вероятность республиканцев", f"{prediction['probability_republican']:.1%}")


st.title("Выборы президента США — 2028")
st.write("Прогноз модели по штатам, вероятности победы и расклад выборщиков.")

try:
    health = requests.get(f"{API_URL}/", timeout=3)
    health.raise_for_status()
    forecasts = load_forecasts(API_URL, ELECTION_YEAR)
except requests.RequestException as error:
    st.error(f"FastAPI недоступен по адресу {API_URL}: {error}")
    st.info("Убедитесь, что API запущен. Для локального запуска задайте API_URL=http://127.0.0.1:8000.")
    st.stop()

totals = electoral_vote_totals(forecasts)
dem_col, rep_col, threshold_col = st.columns(3)
dem_col.metric("Демократы", f"{totals['Democrat']} выборщиков")
rep_col.metric("Республиканцы", f"{totals['Republican']} выборщиков")
threshold_col.metric("Для победы", "270 из 538")

render_forecast_map(forecasts)

state = st.selectbox(
    "Прогноз для выбранного штата",
    options=sorted(STATE_NAMES),
    format_func=lambda code: f"{STATE_NAMES[code]} ({code})",
)
selected_forecast = next(item for item in forecasts if item["state"] == state)
winner_label = "Демократы" if selected_forecast["winner"] == "Democrat" else "Республиканцы"
state_col, probability_col, electoral_col = st.columns(3)
state_col.metric("Прогноз", winner_label)
probability_col.metric(
    f"Вероятность {winner_label.lower()}",
    f"{selected_forecast['probability_democrat'] if selected_forecast['winner'] == 'Democrat' else selected_forecast['probability_republican']:.1%}",
)
electoral_col.metric("Выборщики штата", str(ELECTORAL_VOTES[state]))

render_custom_prediction()
