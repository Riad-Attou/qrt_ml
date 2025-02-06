# Définitions des données

## Données Cliniques 

- **BM_BLAST** (Blastes dans la moelle osseuse) :  
  Les blastes sont des cellules immatures produites dans la moelle osseuse qui, en temps normal, se transforment en cellules sanguines fonctionnelles (globules rouges, globules blancs et plaquettes). Dans la leucémie, un excès de blastes empêche la production normale des cellules du sang.  
  Un taux élevé de blastes est un indicateur clé de la sévérité de la leucémie et peut influencer les prédictions de survie.

- **WBC** (White Blood Cells - Globules blancs) :  
  Les globules blancs sont les cellules du système immunitaire qui protègent le corps contre les infections. Dans la leucémie, leur nombre peut être soit trop élevé (signe de prolifération anarchique), soit trop bas (affaiblissement du système immunitaire).  
  Une variation importante du nombre de globules blancs est un indicateur du dysfonctionnement de la moelle osseuse et peut être corrélée à la progression de la maladie.

- **ANC** (Absolute Neutrophil Count - Neutrophiles absolus) :  
  Les neutrophiles sont un type de globules blancs essentiels pour la lutte contre les infections bactériennes et fongiques. Une diminution du nombre de neutrophiles (neutropénie) rend les patients vulnérables aux infections graves.  
  Un faible taux de neutrophiles peut être un facteur de risque de complications, influençant ainsi la survie des patients atteints de leucémie.

- **MONOCYTES** (Monocytes sanguins) :  
  Les monocytes sont des cellules immunitaires qui participent à l’élimination des agents pathogènes et à la régulation de l’inflammation. Un nombre anormal de monocytes peut signaler une réaction anormale du système immunitaire liée à la leucémie.  
  Les monocytes peuvent être un marqueur de l’activité leucémique et aider à affiner les modèles de prédiction de survie.

- **HB** (Hémoglobine) :  
  L’hémoglobine est une protéine contenue dans les globules rouges qui permet le transport de l’oxygène dans le sang. Dans la leucémie, une baisse du taux d’hémoglobine (anémie) peut survenir en raison de la diminution de la production de globules rouges.  
  Une anémie sévère peut être un facteur influençant le pronostic des patients, en particulier en termes de fatigue et de complications associées.

- **PLT** (Plaquettes sanguines) :  
  Les plaquettes sont des cellules qui permettent la coagulation du sang et préviennent les hémorragies. Dans la leucémie, leur nombre peut être réduit (thrombopénie), entraînant des risques de saignements anormaux.  
  Un faible taux de plaquettes peut être un indicateur de gravité de la maladie et affecter la probabilité de survie.

- **CYTOGENETICS** (Caryotype cytogénétique du patient) :  
  L’analyse cytogénétique consiste à étudier les anomalies des chromosomes des cellules cancéreuses. Certaines modifications spécifiques, comme la perte d’un chromosome (monosomie) ou la présence d’une translocation génétique, sont associées à un pronostic plus ou moins favorable.  
  Les anomalies chromosomiques sont des marqueurs majeurs de la progression de la leucémie et permettent d’améliorer la précision des modèles prédictifs.

---

## Données Moléculaires Génétiques

- **CHR, START, END** (Localisation chromosomique de la mutation) :  
  Ces informations indiquent la position exacte d’une mutation dans le génome humain, c’est-à-dire sur quel chromosome et à quel endroit précis la modification génétique s’est produite.  
  Localiser les mutations permet d’identifier les régions du génome les plus affectées par la leucémie et d’améliorer l’interprétation des facteurs génétiques influençant la survie.

- **REF, ALT** (Mutation nucléotidique) :  
  "REF" correspond à la séquence normale d’ADN, tandis que "ALT" représente la version mutée. Une mutation peut altérer l’activité normale d’un gène et favoriser la prolifération des cellules cancéreuses.  
  Comprendre quelles mutations sont présentes aide à identifier les mécanismes biologiques impliqués dans la maladie et à développer des modèles plus précis.

- **GENE** (Nom du gène muté) :  
  Chaque gène code pour une protéine qui joue un rôle précis dans l’organisme. Certains gènes sont directement impliqués dans la croissance des cellules et peuvent devenir hyperactifs ou inactifs à cause d’une mutation.  
  Certains gènes sont connus pour être des marqueurs de mauvais pronostic en leucémie, et leur analyse peut améliorer la précision du modèle prédictif.

  - **FLT3** : Un gène souvent muté dans la leucémie myéloïde aiguë, associé à une forme plus agressive de la maladie.  
  - **TP53** : Un gène suppresseur de tumeurs qui, lorsqu’il est muté, empêche la cellule de réparer son ADN endommagé et favorise donc la prolifération des cellules cancéreuses.

- **PROTEIN_CHANGE** (Modification de la protéine) :  
  Une mutation dans un gène peut entraîner une modification de la protéine qu’il code, altérant ainsi son fonctionnement normal.  
  Les mutations qui modifient fortement une protéine clé dans la régulation des cellules peuvent être des indicateurs de la gravité du cancer.

- **EFFECT** (Effet de la mutation sur la fonction du gène) :  
  Une mutation peut avoir différents effets : elle peut être bénigne, provoquer une perte de fonction du gène ou, au contraire, activer un gène qui favorise la prolifération cellulaire.  
  Connaître l’effet d’une mutation permet de distinguer les mutations critiques de celles sans impact majeur.

- **VAF** (Variant Allele Frequency - Fréquence allélique de la mutation) :  
  Ce pourcentage indique combien de cellules contiennent la mutation par rapport au total des cellules analysées. Une valeur élevée signifie que la mutation est très répandue parmi les cellules cancéreuses.  
 Un VAF élevé peut être un indicateur de la progression du cancer et influencer les prédictions de survie.