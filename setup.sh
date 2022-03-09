mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
[theme]\n\
base="dark"\n\
primaryColor="#d33682"\n\
backgroundColor="#0E1117"\n\
secondaryBackgroundColor="#31333F"\n\
font=“sans serif”\n\
" > ~/.streamlit/config.toml
