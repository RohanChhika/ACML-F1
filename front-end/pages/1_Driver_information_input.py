import streamlit as st

# Dictionary of drivers
driver_names = {
    1: 'Max VERSTAPPEN', 2: 'Logan SARGEANT', 4: 'Lando NORRIS',
    10: 'Pierre GASLY', 11: 'Sergio PEREZ', 14: 'Fernando ALONSO',
    16: 'Charles LECLERC', 20: 'Kevin MAGNUSSEN', 21: 'Maxwell ESTERSON',
    22: 'Yuki TSUNODA', 24: 'ZHOU Guanyu', 27: 'Nico HULKENBERG',
    31: 'Esteban OCON', 44: 'Lewis HAMILTON', 55: 'Carlos SAINZ',
    63: 'George RUSSELL', 81: 'Oscar PIASTRI', 23: 'Alexander ALBON',
    34: 'Felipe DRUGOVICH', 77: 'Valtteri BOTTAS', 18: 'Lance STROLL',
    3: 'Daniel RICCIARDO', 5: 'Gabriel BORTOLETO', 6: 'Isack HADJAR',
    7: 'Jack DOOHAN', 8: 'Gregoire SAUCY', 9: 'Nikola TSOLOV',
    12: 'Kimi ANTONELLI', 15: 'Gabriele MINI', 17: 'Caio COLLET',
    19: 'Tommy SMITH', 25: 'Hugh BARTER', 26: 'Nikita BEDRIN',
    28: 'Ryo HIRAKAWA', 29: "Patricio O'WARD", 30: 'Liam LAWSON',
    39: 'Arthur LECLERC', 40: 'Ayumu IWASA', 41: 'Isack HADJAR',
    42: 'Frederik VESTI', 50: 'Ryo HIRAKAWA', 61: 'Jack DOOHAN',
    98: 'Theo POURCHAIRE', 36: 'Jake DENNIS', 37: 'Ayumu IWASA',
    45: 'Franco COLAPINTO', 38: 'Dino BEGANOVIC', 97: 'Robert SHWARTZMAN',
    43: 'Franco COLAPINTO', 46: 'Luke BROWNING', 87: 'Oliver BEARMAN',
    62: 'Ryo HIRAKAWA', 72: 'Frederik VESTI'
}

name_to_number = {v: k for k, v in driver_names.items()}

def race_driver_setup():
    st.title("Race Driver Setup")

    st.markdown("### Select Participating Drivers")
    selected_drivers = st.multiselect(
        "Choose drivers for the race:",
        options=list(driver_names.values())
    )

    st.markdown("---")
    if selected_drivers:
        st.markdown("### Enter Qualifying Positions and Lap Times")
        race_data = []
        for driver in selected_drivers:
            col1, col2 = st.columns([1, 1.5])
            with col1:
                pos = st.number_input(f"Qualifying Position - {driver}", min_value=1, max_value=100, step=1, key=f"pos_{driver}")
            with col2:
                lap_time = st.text_input(f"Lap Time (e.g., 1:32.456) - {driver}", key=f"lap_{driver}")
            race_data.append({
                "quali_position": pos,
                "quali_lap_time": lap_time,
                "driver_number": name_to_number.get(driver)
            })
        
        st.markdown("---")
        st.subheader("Summary")
        st.dataframe(race_data)
        return race_data

    else:
        st.info("Please select at least one driver to input race data.")

st.session_state["race_data"] = race_driver_setup()
