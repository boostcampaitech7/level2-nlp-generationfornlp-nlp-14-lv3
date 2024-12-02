apt-get update
apt-get install wget
apt install curl -y
apt install -y git
curl -fsSL https://ollama.com/install.sh | sh
python -m venv level2
source level2/bin/activate 
pip install -r ./requirements.txt


echo "====================================="
echo "Environment setup complete!"
echo "To start using Ollama, please:"
echo "1. Open a new terminal and run: ollama serve"
echo "2. Return to original terminal and run:"
echo "   ollama pull aya-expanse:32b"
echo "   ollama pull aya-expanse:8b"
echo "   ollama pull granite3-dense:8b"
echo "   ollama pull granite3-dense:2b"
echo "====================================="