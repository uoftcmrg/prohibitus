# Prohibitus

Music Generation AI by Juho Kim, Minchan Kim, and Nikolas Marinkovich

## Requirements

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [PrettyMIDI](https://github.com/craffel/pretty-midi)

## Guides

### Generate in ABC file format

2. Download mscz music samples

Example:

```shell
python ./download.py ./mscz-files-small.csv ../scores/mscz
```

3. Convert mscz to xml

Example:

```shell
python ./mscz2xml.py ../scores/mscz/ ../scores/xml/ "C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
```

4. Convert xml to abc

Example:

```shell
python xml2abc.py -o ../scores/abc/ ../scores/xml/*.xml
```

## Credits

- Inspiration
    - [karpathy: mingpt](https://github.com/karpathy/minGPT)
- Music score file format conversion
    - [Willem Vree: xml2abc.exe](https://wim.vree.org/svgParse/xml2abc.html)
    - [Willem Vree: abc2xml.exe](https://wim.vree.org/svgParse/abc2xml.html)
    - [MuseScore](https://musescore.org/)
- Datasets
    - [Xmader: MuseScore dataset](https://github.com/Xmader/musescore-dataset)
    - [asigalov61: Tegridy-MIDI-Dataset](https://github.com/asigalov61/Tegridy-MIDI-Dataset)
    - [fredrik-johansson: midi](https://github.com/fredrik-johansson/midi)
    - [Classical Archives - The Greats (MIDI)](https://thepiratebay.org/description.php?id=6734800)

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
