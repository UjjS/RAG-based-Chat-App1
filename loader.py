import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_from_url(url: str):
    """Fetches and extracts clean text from a URL."""
    try:
        response = requests.get(url)  # URL se data liya
        response.raise_for_status()  # Agar status kharab hai toh exception uthao
        soup = BeautifulSoup(response.text, 'html.parser')  # HTML ko parse kiya
        
        # script aur style elements hata diye
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Text nikala aur saaf kiya
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())  # Har line ko trim kiya
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # Double space se split kiya
        text = "\n".join(chunk for chunk in chunks if chunk)  # Khaali lines hata di
        
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")  # Error aayi toh print kiya
        return None

def get_text_chunks(text: str):
    """Splits a long text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Har chunk ki max size
        chunk_overlap=200,  # Chunk ke beech overlap
        length_function=len  # Length nikalne ke liye len use kiya
    )
    chunks = text_splitter.split_text(text)  # Text ko chote parts mai baant diya
    return chunks