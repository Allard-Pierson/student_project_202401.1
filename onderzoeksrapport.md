# Onderzoeksrapport

## Voorwoord
Met genoegen presenteren wij het resultaat van ons gezamenlijke onderzoek, uitgevoerd in het kader van de minor Applied Artificial Intelligence aan de Hogeschool van Amsterdam, in samenwerking met de Universiteitsbibliotheek van Amsterdam. Dit onderzoek is het product van de toegewijde inspanningen van ons team bestaande uit vijf studenten, die gezamenlijk hebben gewerkt aan het ontwikkelen van een toepassing dat in staat is een zoekvraag te interpreteren en te koppelen aan een classificatiesysteem van de bibliotheek van de UvA/HvA.

Dit rapport biedt een gedetailleerd overzicht van ons onderzoeksproces. Wij bedanken bij deze Jaroen Kuijpers, Marjolein Beumer en David van Dijk voor de ondersteuning die ons geboden is de afgelopen paar maanden tijdens het gehele proces.

## Inhoudsopgave

- Inleiding
  - Probleemstelling
  - Scope
  - Begrippenlijst

- Onderzoeksresultaten & Analyse
  - Annif
    - Dataset
    - Modellen
  - Alternatieve modellen
    - BERT
    - ROBERTa
    - Naive Bayes
    - Support Vector Machine (SVM)

- Conclusie & Aanbevelingen

## Inleiding
### Probleemstelling
Voor de minor Applied Artificial Intelligence (AAI) is er een opdracht opgesteld door de Universiteitsbibliotheek van Amsterdam om een toepassing te ontwikkelen dat een zoekvraag interpreteert en kan verbinden aan het classificatiesysteem die de UB hanteert. Hiermee is het einddoel van de UB om een soort zoekassistent te ontwikkelen die het classificatiesysteem begrijpt en om zo de zoek functionaliteit te vebeteren. En om daarmee de toegankelijkheid van de omvangrijke informatiebronnen in de bibliotheek te vergroten.

Tijdens het project is duidelijk geworden dat de hierboven genoemde taak een grote uitdaging was, waardoor er een afweging is gemaakt om een eerste stap te zetten richting dit doel.
Deze eerste stap van het proces is om classificatie codes te verkrijgen vanuit een zoekvraag d.m.v. kunstmatige intelligentie. Dit is dan ook waar wij ons mee bezig hebben gehouden tijdens de duratie van dit project.

Voor het uitwerken van deze taak is onderzoek gedaan naar de tool Annif en hebben ook onderzoek gedaan naar alternatieve methodes. 

### Scope
Er zijn een aantal classificatiesystemen die gehanteerd worden binnen de bibliotheek.
Om de scope van de opdracht te verkleinen hebben wij een focus gelegd op het classificatie systeem van de Library of Congress (afgekort als LCC).

### Begrippenlijst

- **Library of Congress Classification plaatscode**: Een codesysteem gebruikt door bibliotheken om materialen te classificeren op onderwerp. Het is ontwikkeld door de Library of Congress en wordt vaak gebruikt om boeken en andere materialen te organiseren. Een plaatscode geeft dus de locatie van bijvoorbeeld een boek.

- **Annif**: Een geautomatiseerde toolkit voor onderwerpsindexering.

- **MARC XML data**: Machine-Readable Cataloging in XML, een gestandaardiseerd formaat voor het coderen van bibliografische gegevens in XML.

- **Library of Congress Subject Headings (LCSH)**: Een thesaurus van onderwerpstitels bijgehouden door de Library of Congress, gebruikt om onderwerpen te identificeren en te classificeren in bibliotheekcatalogi.

- **Extreme multi-label classificatie**: Dit is een supervised learning probleem, waar een input geassocieerd wordt met meerdere labels. Dit kenmerkt zich door het grote verschil- en aantal labels waarop het geclassificeerd moet worden. Subject indexing is hiervan een voorbeeld. 

- **Natural Language Processing (NLP)**: Het is een tak van kunstmatige intelligentie die zich bezighoudt met het begrijpen, interpreteren en genereren van menselijke taal door computers. Het doel van NLP is om computers in staat te stellen menselijke taal op een zinvolle manier te begrijpen en erop te reageren.

- **Large Language Model (LLM)**: Een geavanceerd taalmodel dat gebruikmaakt van kunstmatige intelligentie en machine learning om natuurlijke taal te begrijpen en te genereren op grote schaal. Denk hierbij aan ChatGPT

- **Conversational search**: Conversational search verwijst naar het proces van het zoeken naar informatie op het internet door middel van natuurlijke gesprekken met zoekmachines. In plaats van traditionele zoekopdrachten bestaande uit losse trefwoorden.

## Onderzoeksresultaten & Analyse

### Annif
Voor dit project was ons doel om een [Library of Congress Classification plaatscode](https://www.library.northwestern.edu/find-borrow-request/catalogs-search-tools/understanding-call-numbers/loc.html) te genereren voor een zoekopdracht. Een optie die we daarvoor hebben onderzocht is het gebruik van de tool [Annif](https://github.com/NatLibFi/Annif?tab=readme-ov-file).

Annif wordt als volgt gedefinieerd door de makers:<br>
"Annif is een geautomatiseerd toolkit voor onderwerpsindexering. Het werd oorspronkelijk gemaakt als een statistisch geautomatiseerd indexeringstool dat metadata van de Finna.fi gebruikte als trainingscorpus."

Dit betekent dat Annif in staat is onderwerpen te genereren die zijn verbonden met een specifieke vocabulaire of classificatie door metadata van een bibliotheek te gebruiken als trainingscorpus.

Hoe hebben we Annif geïmplementeerd? Aangezien we uiteindelijk de LCC plaatscodes wilden verkrijgen, was onze eerste stap om dit te proberen te implementeren in Annif. Deze tool biedt ons al een manier om onderwerpen te genereren met behulp van een specifieke vocabulair. Het enige probleem was dat het geen implementatie had voor de Library of Congress Classification. Daarom moesten we een manier vinden om deze classificatie in de tool te laden.

Na wat onderzoek is gebleken dat het Library of Congress Classification-schema niet volledig beschikbaar is. Daarom hebben we als alternatief gekozen voor [Library of Congress Subject Headings](https://id.loc.gov/authorities/subjects.html) of LCSH.<br>
Library of Congress Subject Headings is, zoals de naam al aangeeft, een thesaurus van onderwerpstitels die bijgehouden wordt door de Library of Congress. Hoewel het een apart systeem is van de Library of Congress Classification, overlapt het op sommige manieren. Veel onderwerpen binnen LCSH bevatten bijvoorbeeld een plaatscode of een specifiek bereik van plaatcodes waarin een onderwerp zich bevindt. Hiermee kunnen deze gegenereerde onderwerpen dus gebruikt worden om tot een plaatscode te komen.

Bijvoorbeeld hebben we het onderwerp [__Learning models (Stochastic processes)__](https://id.loc.gov/authorities/subjects/sh85075543.html). Dit onderwerp kan door Annif als de beste match voor onze zoekopdracht worden gekozen. Dit specifieke onderwerp bevat de LCC-plaatscode: __QA274.6__. Op deze manier kiezen we het onderwerp met de hoogste score voor onze zoekopdracht en kunnen we de plaatscodes uit de URI van de Library of Congress Subject Heading halen.

Nu de algemene werking van de implementatie is genoemd, gaan we iets dieper in op de oorsprong van de dataset en de modellen.

#### Dataset
Als eerst hebben we de dataset. Hoe deze dataset is gemaakt, wordt gedetailleerd beschreven in het notebook [data_processing.ipynb](api/annif/data_processing.ipynb).

De gegevens die we hebben gebruikt om een model voor Annif te trainen, zijn de [MARC XML data van de Universiteitsbibliotheek van Amsterdam](https://uba.uva.nl/ondersteuning/open-data/datasets-en-publicatiekanalen/datasets-en-publicatiekanalen.html#Boeken) en de [MARC XML-gegevens van Springer Nature](https://metadata.springernature.com/metadata/books). 

Hier richten we ons op de metadata van de boeken en alleen op Engelstalige gegevens. Dit is om de scope van de implementatie te beperken. Uit alle records worden alleen de records gebruikt die één of meer LCSH-onderwerpen bevatten.

De Amsterdamse Universiteitsbibliotheek publiceert hun data ook in andere formaten, zoals het nieuwere LOD. Wij hebben er voor gekozen om gebruik te maken van de MARC-XML data voor het maken van de dataset, aangezien deze makkelijker was om te ontleden in Python. Ook is er gebruik gemaakt van extra metadata vanuit Springer Nature. Deze data bevat namelijk veel Engelstalige records met LCSH-onderwerpen. De reden voor het gebruik van deze data is omdat er niet genoeg records waren vanuit de UB die aan de eerder genoemde eisen voldeden, waardoor er niet getraind kon worden op een groot aantal onderwerpen. Door deze twee bronnen te combineren is er wel voldoende data om een goed model te trainen.
 
Uiteindelijk zijn er twee aparte datasets gemaakt, één met alleen titels als trainingscorpus en een andere met zowel titels, als samenvattingen als trainingscorpus. Deze scheiding is gemaakt omdat sommige modellen beter trainen op kleinere teksten en andere op grotere teksten.
De gegevens worden geëxtraheerd en verwerkt op een manier die bruikbaar is door Annif.

#### Modellen
Annif biedt een reeks modellen die al zijn geïmplementeerd in de tool, de volledige lijst kan worden bekeken op hun [wikipagina](https://github.com/NatLibFi/Annif/wiki). Van deze modellen hebben we enkele geïmplementeerd, en ook een paar eigen toegevoegd. Deze zullen hieronder worden genoemd. Voor een volledige uitwerking hoe deze modellen zijn getraind en geëvalueerd, bekijk dan het notebook [make_project.ipynb](api/annif/make_project.ipynb). 

##### TF-IDF
Staat voor _Term Frequency Inverse Document Frequency_. Het combineert twee scores

- Term frequency: De frequentie van een woord in een document, dus hoe vaak het voorkomt.
- Inverse document frequency: Geeft aan hoe veel een woord voorkomt in alle documenten.

Dit is een redelijk simpel machine learning model, 

**Toepassing op de dataset:**: Er zijn twee modellen van TF-IDF getraind. Eén hiervan is getraind op alleen de titels als tekstcorpus en de andere op de volledige samenvattingen. De labels die geclassificeerd moeten worden zijn de onderwerpen, in dit geval in de vorm van URI's (Dit geldt voor elk model in Annif).

##### Omikuji Parabel
[Omikuji](https://github.com/tomtung/omikuji) Parabel is een implementatie van van tree-based modellen, denk hierbij aan _Decision Tree_ of _Random Forest_. Deze implementatie verschilt in het feit dat het gebruikt wordt voor extreme multi-label classificatie. 

**Toepassing op de dataset**: Voor dit model is gebruik gemaakt van de dataset met samenvattingen als tekstcorpus.

##### XTransformer
[XTransformer](https://github.com/amzn/pecos/blob/mainline/pecos/xmc/xtransformer/README.md) is een zelf toegevoegde backend in Annif. Deze maakt gebruik van het PECOS framework om extreme multi-label classificatie toe te passen d.m.v. transformer modellen. In dit geval wordt DistilBERT gebruikt, wat een verkleinde versie is van het meer bekende BERT, waar later meer over verteld wordt.

**Toepassing op de dataset**: Voor dit model is gebruik gemaakt van de dataset met samenvattingen als tekstcorpus. Wel is het aantal rijen aan data sterkt verminderd naar 50.000, aangezien dit anders te intensief was op te trainen.

##### Decision Tree
Een ander model wat wij zelf hebben toegevoegd is een decision tree.
Een decision tree, of beslissingsboom, is een machine learning-algoritme die werk d.m.v. een boomachtige structuur waarin elke knoop een beslissing vertegenwoordigt op basis van een bepaald kenmerk van de inputdata. De bladeren van de boom geven de uiteindelijke uitkomst of voorspelling weer.

Bij het trainen van een decision tree wordt de dataset opgesplitst op basis van verschillende kenmerken, waarbij het doel is om de data in homogene subsets te verdelen. Deze splitsingen worden genomen op basis van criteria zoals entropie, die de mate van homogeniteit in de resulterende subsets meten.

**Toepassing op de dataset voor decision tree:**
Voor dit model is gebruik gemaakt van de dataset met samenvattingen als tekstcorpus.

Hoewel we dit model hebben uitgewerkt, heeft het niet de gewenste resultaten opgeleverd. Deze wordt hier gemeld ter volledigheid van het onderzoek, maar deze wordt niet meegenomen in de evaluatie.

#### Evaluatie
Tijdens evaluatie worden tal van metrics geproduceerd, waaronder [precision, recall](https://en.wikipedia.org/wiki/Precision_and_recall) en [F1-scores](https://en.wikipedia.org/wiki/F-score).

Er is ook een andere belangrijke metric, genaamd [NDCG](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0), wat staat voor _Normalized Discounted Cumulative Gain_. Deze metric wordt gebruikt om de kwaliteit van een rangschikking te meten (vergelijkbaar met de onderwerpen die worden geproduceerd in Annif) en wordt daarom vaak gebruikt om de prestaties van zoekmachines of aanbevelingssystemen te evalueren. Het werkt door het toekennen van een score aan de relevantie van elk onderwerp ten opzichte van de zoekopdracht. Deze scores worden vervolgens aangepast op basis van de positie in de zoekresultaten. Omdat dit een goede meting is voor onze taak in Annif, zullen we voornamelijk naar deze metric kijken tijdens de evaluatie van de modellen.

De resultaten zijn als volgt. Hierbij wordt de NDCG score gegeven als een getal tussen de nul en één, waarbij een hogere waarde een indicatie geeft van een betere prestatie.

| Model | NDCG (0 - 1)| 
|----------|----------|
| TF-IDF (Titels) | 0.215 |
| TF-IDF | 0.357 |
| Omikuji Parabel | 0.587 |
| XTransformer| 0.487 |

Hieruit is dus te concluderen dat  Omikuji de beste uitkomst geeft, wel is het mogelijk om te kijken naar implementaties om de XTransformer op een volledige trainingscorpus te laten trainen, hierdoor kan de prestatie van dit model wellicht verbeteren.

### Alternatieve modellen
We hebben dus veel onderzoek gedaan naar het implementeren van Annif voor onze opdracht. Naast die tool is er ook onderzoek gedaan naar alternatieve manieren om plaatscodes te kunnen genereren. Dit hebben wij gedaan om een oplossing te kunnen vinden als alternatief op Annif. Hiervoor hebben we een aantal losstaande classificatie modellen toegepast voor multi-label classificatie. Hieronder een overzicht. De uitwerking van deze modellen staat in de volgende [notebooks](alternatieve_modellen).

#### BERT

Ten eerste hebben we het model BERT geprobeerd. 

BERT is een model voor natuurlijke taalverwerking dat is ontwikkeld door Google. Het staat voor Bidirectional Encoder Representations from Transformers. BERT kan complexere patronen en afhankelijkheden in taal vast te leggen, wat leidt tot betere prestaties in taken zoals tekstclassificatie, herkenning van vernoemde entiteiten, vraagbeantwoording en meer. Het model is vooraf getraind op grote hoeveelheden tekst uit diverse bronnen.

**Toepassing op de dataset:**
Voor Bert is er gebruik gemaakt van de dataset met samenvattingen van boeken van de bibliothkeek met het daarbij horende genre.
Het idee was om het Bert model te trainen op de samenvattingen om zo het bijbehorende genre te kunnen voorspellen. Dit is gedaan door het vooraf getrainde model aan te passen en te fine-tunen op de dataset.

#### RoBERTa

Vervolgens hebben we het model RoBERTa geprobeerd.

RoBERTa staat voor Robustly optimized BERT approach. Dit model is gebaseerd op het BERT-framework. RoBERTa is ontwikkeld door Facebook AI Research. Het doel van dit model is om de tekortkomingen van BERT aan te pakken en de prestaties op verschillende NLP-taken verder te verbeteren. RoBERTa gebruikt verschillende optimalisaties in de trainingsprocedure ten opzichte van BERT. 

**Toepassing op de dataset:**
De code voor het fine-tunen van het RoBERTa-model is op dezelfde manier opgezet als de code voor het BERT-model. Het vooraf getrainde model wordt ingeladen en gefinetuned op de dataset.

#### Naive Bayes

Ook hebben we Naive Bayes geprobeerd.

Naive Bayes is een probabilistisch classificatiemodel dat gebaseerd is op Bayesiaanse statistieken en de veronderstelling van conditional independence tussen de functies. Het wordt "naïef" genoemd omdat het de aanname maakt dat alle kenmerken onafhankelijk zijn van elkaar, wat niet altijd het geval is in de werkelijkheid. Desondanks heeft Naive Bayes bewezen effectief te zijn in veel tekstclassificatietaken.

**Toepassing op de dataset:**
Voor Naive Bayes is er gebruikgemaakt van een dataset met samenvattingen van boeken van de bibliotheek, samen met de bijbehorende genres. Het doel was om het Naive Bayes-model te trainen op deze samenvattingen om zo het bijbehorende genre te kunnen voorspellen. Door de probabilistische aard van Naive Bayes kan het model de kans berekenen dat een bepaalde samenvatting tot een bepaald genre behoort, wat nuttig is voor tekstclassificatie.

#### Support Vector Machine (SVM)

Tot slot hebben we nog SVM geprobeerd. 

Support Vector Machine (SVM) is een krachtig supervised machine learning-model dat wordt gebruikt voor zowel classificatie als regressie. SVM zoekt naar een optimale hyperplane in de feature space om de data te scheiden in verschillende klassen. Het doel is om een hyperplane te vinden dat de afstand tot de dichtstbijzijnde datapunten van beide klassen maximaliseert.

**Toepassing op de dataset:**
Voor SVM is er gebruikgemaakt van een dataset met boeksamenvattingen en bijbehorende genres. Het SVM-model is getraind op deze dataset om de samenvattingen in verschillende genres te classificeren. SVM is effectief in het vinden van complexe beslissingsgrenzen, wat het nuttig maakt voor taken zoals tekstclassificatie. Door het vinden van een optimale hyperplane kan het SVM-model goed omgaan met niet-lineaire verbanden tussen features en genres.

## Conclusie & Aanbevelingen

Voor ons project hebben wij dus toepassingen onderzocht om een zoekvraag om te zetten naar een LCC plaatscode. Hiervoor hebben wij onderzoek naar Annif en eigen modellen.

Hieruit kunnen we concluderen dat het gebruik van de tool Annif de beste uitkomst heeft geleverd. Het maken van een eigen implementatie d.m.v. losstaande modellen heeft niet tot een uitkomst geleid waarmee deze gebruikt kon worden om LCC plaatscodes te verkrijgen. Vandaar dat wij voor ons product de focus hebben gelegd bij het implementeren van Annif voor deze taak. Voor onze implementatie van Annif hebben we ander vocabulair ingeladen in de tool, namelijk Library of Congress Subject Headings. Door middel van metadata van de bibliotheek en externe bronnen hebben we hiervoor een specifieke dataset gemaakt. Deze dataset is gebruikt om een aantal modellen te trainen binnen Annif die onderwerpen kunnen voorspellen uit deze vocabulair. Omdat LCSH onderwerpen verbonden zijn met LCC, kunnen we uit deze LCSH onderwerpen de gewenste LCC plaatscodes verkrijgen. Dit is dan ook de uiteindelijke implementatie die gebruikt is voor het product.

De aanbeveling die het team doet voor mensen die dit project voortzetten, is 
om verder onderzoek te doen naar conversational search. Hier zijn wij als team namelijk niet aan toegekomen, terwijl dit wel een groot deel is van de uiteindelijke visie. Wij hopen met de opzet van dit project een beeld te kunnen geven hoe kunstmatige intelligentie gebruikt kan worden om aan plaatscodes te komen. Hierbij duiden we op de implementatie van Annif. Zo kunnen de gegenereerde plaatscodes wellicht gecombineerd worden met bestaande LLM's, zoals ChatGPT om te zorgen voor conversational search. Onze alternatieve implementaties hebben tot niet veel resultaat geleid, vandaar dat het door ons wordt aangeraden om daar niet mee verder te gaan.
