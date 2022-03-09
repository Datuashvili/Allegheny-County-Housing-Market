mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = ‘#08424f’
backgroundColor = ‘#08424f’
secondaryBackgroundColor = ‘#0d5a6a’
textColor= ‘#ffffff"’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
