# Prohibitus

Music Generation AI by Juho Kim, Minchan Kim, and Nikolas Marinkovich

## Guide

1. Generate dataset

    1. Download mscz music samples

   Example:
    ```shell
    python ./download.py ./mscz-files-small.csv ../scores/mscz
    ```

    2. Convert mscz to xml

   Example:
    ```shell
    python ./mscz2xml.py ../scores/mscz/ ../scores/xml/ "C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
    ```

    3. Convert xml to abc

   Example:
    ```shell
    python xml2abc.py -o ../scores/abc/ ../scores/xml/*.xml
    ```

## Credits

- karpathy
  - mingpt: [minimalist GPT implementation](https://github.com/karpathy/minGPT)
- Xmader
  - mscz-files.csv: [MuseScore dataset](https://github.com/Xmader/musescore-dataset)
- Willem Vree
    - xml2abc.py: MusicXML to ABC conversion
    - abc2xml.py: ABC to MusicXML conversion

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
