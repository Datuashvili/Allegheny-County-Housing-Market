mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
[theme]\n\
primaryColor="#08424f"\n\
backgroundColor="#08424f"\n\
secondaryBackgroundColor="#0d5a6a"\n\
textColor="#ffffff"\n\
font=“sans serif”\n\
\n\
" > ~/.streamlit/config.toml
