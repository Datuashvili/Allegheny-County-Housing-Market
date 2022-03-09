mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = ‘#F63366’
backgroundColor = ‘0E1117’
secondaryBackgroundColor = ‘#31333F’
textColor= ‘#FAFAFA"’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
