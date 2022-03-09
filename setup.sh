mkdir -p ~/.streamlit/

echo "\
[theme]\n\
primaryColor = ‘#F63366’\n\
backgroundColor = ‘0E1117’\n\
secondaryBackgroundColor = ‘#31333F’\n\
textColor= ‘#FAFAFA"’\n\
font = ‘sans serif’\n\
" > ~/.streamlit/config.toml


echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
