import streamlit as st

# Set page configuration

# Set up page title and icon
st.set_page_config(page_title="F1 Predictor", page_icon="🏎️")

# Custom styling with red theme
st.markdown(
    """
    <style>
    .main {
        background-color: #fdf6f6;
    }
    h1 {
        color: #fff;
        font-size: 3em;
    }
    .intro-text {
        font-size: 1.2em;
        line-height: 1.6;
        color: #fff;
    }
    .highlight {
        color: #d62828;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Content
st.title("🏁 Welcome to the F1 Predictor")
st.markdown(
    """
    <div class="intro-text">
        The <span class="highlight">F1 Predictor</span> is an interactive tool that lets you explore and predict Formula 1 race outcomes 
        based on historical data and machine learning models. Whether you're a hardcore fan, a data enthusiast, or just curious, 
        our app gives you the power to simulate race results using real-world stats.
        <br><br>
        Here's what you can do:
        <ul>
            <li><b>Input driver and team data</b> to generate race predictions</li>
            <li><b>Explore model accuracy</b> and performance graphs</li>
        </ul>
        Built with ❤️ using Python, Pandas, and Streamlit.
        <br><br>
        Use the sidebar to navigate to prediction tools and data exploration pages.
    </div>
    """,
    unsafe_allow_html=True
)
