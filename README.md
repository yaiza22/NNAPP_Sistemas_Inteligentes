# Autores
> Yaiza Angelina Sanchez Dueñez - 2220232034

> Nicolas Gonzalez Ortiz - 2220261089

# Install the virtual environment
> python -m venv .venv

# Activate the venv in Windows
> .\.venv\Scripts\activate

# Activate the venv in Linux
> source .venv\bin\activate

# install requirements packages
> pip install -r .\requirements.txt

# Abre un terminal y ejecutas el Backend:
> uvicorn app.main:app --reload --port 8000

# Abre otro terminal y ejecutas el frontned
> streamlit run ui/app.py
