mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
[theme]\n\
base="dark"\n\
primaryColor="#d33682"\n\
backgroundColor="#002b36"\n\
secondaryBackgroundColor="#586e75"\n\
font=“sans serif”\n\
" > ~/.streamlit/config.toml
