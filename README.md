<!-- Improved compatibility of back to top link: See: https://github.com/Ayenem/LDS -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
<!-- [![Forks][forks-shield]][forks-url] -->
<!-- [![Stargazers][stars-shield]][stars-url] -->
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://datawow.io/blogs/text-summarisation-by-textran">
    <img src="images/summarization.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Long Document Summarization with TextRank</h3>

  <p align="center">
    Automatic summarization with semi-automatic pre-processing of long documents
    <br />
    <br />
    <a href=#usage>View Demo</a>
    ·
    <a href="https://github.com/Ayenem/LDS/issues">Report Bug</a>
    ·
    <a href="https://github.com/Ayenem/LDS/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#references">References</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://www.researchgate.net/publication/232645575_Graph-Based_Algorithms_for_Text_Summarization)

Long Document Summarization (LDS) is a NLP task motivated by source documents where the texts exceed the model’s context lengths. As there is no commonly agreed-upon solution to this problem, LDS remains an active research area (<a href="#references">Tunstall et al., 2022</a>).

<a href="#references">Vig et al. (2021)</a> report two-step extractive-abstractive frameworks as a main category for approaching the Long Document Summarization (LDS) task. This approach consists of:

1. Extracting a subset of the text.
    * Here-applied by a regex-based approach for automatic segmentation, reduction and cleaning of the input document.
2. Feeding it to an abstractive [or extractive] summarization model.
    * You may:
        1. Use one of the re-imported HuggingFace abstractive summarization models
        2. Use the provided stand-alone implementation of TextRank (Mihalcea and Tarau, 2004)
        3. Pipe your own summarization model

<!-- REFERENCES -->
## References

Lewis Tunstall, Leandro von Werra, and Thomas Wolf. 2022. *Natural language processing with transformers*, chapter 6. "O'Reilly Media, Inc.".

Jesse Vig, Alexander R Fabbri, and Wojciech Kryściński. 2021. Exploring neural models for query-focused summarization. *arXiv preprint arXiv:2112.07637.*

Rada Mihalcea and Paul Tarau. 2004. TextRank: Bringing Order into Text. In *Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing*, pages 404-411, Barcelona, Spain. Association for Computational Linguistics.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

[Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
### Installation

```sh
git clone https://github.com/Ayenem/LDS.git
cd LDS/
poetry install
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### **I just want to use TextRank for extractive summarization**

```py
from LDS.textrank import TextRank
from sentence_transformers import SentenceTransformer

summarizer = TextRank(
    sentence_encoder = SentenceTransformer("Sahajtomar/french_semantic"),
)
summary = summarizer(text, n_sentences=5)
print(summary)
```

### **I want to read and pre-process my document(s) first**

```py
from LDS.book_loader import BookLoader
from LDS.textrank import TextRank
from sentence_transformers import SentenceTransformer

book = BookLoader(
    doc_path = "data/D5627-Dolan.docx", # Word documents are handled
    markers  = {                        # Refer to the BookLoader class docstrings for the role of markers
        "slice": [r"^Introduction$", r"Annexe /$"],
        "chapter": r"^Chapitre \d+ /$|^Conclusion$",
        "headers": r"^Chapitre \d+ /.+"
                   r"|^Introduction$"
                   r"|^Stress, santé et performance au travail$"
                   r"|^Conclusion$",
        "footnotes": re.compile(
            r""".+?[A-Z]\.              # At least one character + a capital letter + a dot
                \s.*?                   # + Whitespace + any # of characters
                \(\d{4}\)               # + 4 digits within parens
            """, re.VERBOSE),           # e.g. "12	Zuckerman, M. (1971). Dimensions of ..."
        "undesirables": re.compile(
            r"""^CONFUCIUS$
                |^Matière à réFlexion$
                |^/\tPost-scriptum$
                |^<www\.pbs\.org/bodyandsoul/218/meditation\.htm>.+?\.$
                |^Source\s:\s
            """, re.VERBOSE),
        "citing": re.compile(
            rf"""((?:{RE_ALPHA}){3,}?)  # Capture at least 3 alphabetic characters
                 \d+                    # + at least one digit
            """, re.VERBOSE),           # e.g. "cited1"
        "na_span": [
            # Starts with this:
            r"^exerCiCe \d\.\d /$",
            # Ends with any of these:
            r"^Chapitre \d+ /$"
            r"|^Conclusion$"
            r"|^Les caractéristiques personnelles\."
            r"|/\tLocus de contrôle$"
            r"|^L'observation de sujets a amené Rotter"
            r"|^Lorsqu'une personne souffre de stress"]
    }
)

chapters_to_summarize = book.get_chapters(1, 3)

summarizer = TextRank(
    sentence_encoder = SentenceTransformer("Sahajtomar/french_semantic"),
)

chapter_summaries = [summarizer(chapter, n_sentences=10)
                     for chapter in chapters_to_summarize]
print(chapter_summaries)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [ ] Write a roadmap

<!-- See the [open issues](https://github.com/Ayenem/LDS/issues) for a full list of proposed features (and known issues). -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Ahmed Moubtahij - [@TheAyenem](https://twitter.com/TheAyenem) - moub.ahmed@hotmail.com

Project Link: [https://github.com/Ayenem/LDS](https://github.com/Ayenem/LDS)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

<!-- AM: MENTION THE NETWORKX AN DEAL LIBRARIES -->
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [funcy](https://github.com/Suor/funcy)
* [deal](https://github.com/life4/deal)
* [networkx](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Ayenem/LDS.svg?style=for-the-badge
[contributors-url]: https://github.com/Ayenem/LDS/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Ayenem/LDS.svg?style=for-the-badge
[forks-url]: https://github.com/Ayenem/LDS/network/members
[stars-shield]: https://img.shields.io/github/stars/Ayenem/LDS.svg?style=for-the-badge
[stars-url]: https://github.com/Ayenem/LDS/stargazers
[issues-shield]: https://img.shields.io/github/issues/Ayenem/LDS.svg?style=for-the-badge
[issues-url]: https://github.com/Ayenem/LDS/issues
[license-shield]: https://img.shields.io/github/license/Ayenem/LDS?style=for-the-badge
[license-url]: https://github.com/Ayenem/LDS/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/ahmed-moubtahij/
[product-screenshot]: images/textrank_graph.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
