echo "Downloading CNN weight"

pip install gdown

gdown --id 1MVSSfLaWVyOjy7CHZZRy5ADgnYdG71ME

echo "Downloading LTC weight"

gdown --id 1y00Qip8d9FAtmaVst1wBae1TiY-yjrW9

mkdir -p weights && mv *.pth weights/