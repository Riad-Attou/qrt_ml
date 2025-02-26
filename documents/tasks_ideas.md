# Notes, idées et tâches

## Explication de la leucémie myéloïde

Pour comprendre la leucémie myéloïde, il faut comprendre comment sont produites les trois principales cellules contenues dans le sang : les plaquettes, les globules blancs et les globules rouges. Ces cellules sont créées dans la moelle osseuse avant d'être transférées dans le sang. Au départ, dans la moelle, les cellules sont naïves (elles n'ont pas encore de "spécialisation"). On les appelle des blastes. Dans un corps sain, ces blastes se spécialisent grâce à différentes enzymes puis sont transférées dans le sang régulièrement pour remplacer les cellules mortes.

Cependant, dans un corps atteint de leucémie myéloïde, on retrouve une quantité anormalement haute de blastes dans la moelle. La leucémie attaque le processus de spécialisation des blastes. C'est pourquoi on retrouve des blastes dans la sang par la suite, ce qui ne devrait pas arriver. Ce dysfonctionnement est corrélé à la mutation de la paire de chromosome 22 (voir chromosome Philadelphie sur Wikipédia). Si cette mutation est présente, le patient est atteint de leucémie myéloïde (la réciproque est à vérifier). Cette leucémie se traduit donc par un taux très faible de plaquettes, globules blancs et globules rouges (cependant, un haut ou bas niveau des ces cellules peut être normal si le patient est atteint d'un virus, par exemple).

D'autres facteurs sont à prendre en compte pour déterminer la gravité (relative) de la leucémie. Notamment l'âge, le sexe, l'IMC, le nombre de globules blancs (à priori plus critique que les autres cellules). La présence d'anémie (manque de globules rouges) peut aussi indiquer une leucémie, mais cela peut seulement être une carence en fer. Enfin, des saignements prolongés indiquent certainement un manque de plaquettes.

## Deux modèles

- XGBoost + modèle de Cox + expertise médicale (a priori plus efficace).

    - XGBoost (Extreme Gradient Boosting) est un algorithme d’apprentissage supervisé qui crée une forêt d’arbres de décision, où chaque nouvel arbre corrige les erreurs des arbres précédents. Il est robuste pour notre quantité de données.

    - Le modèle de Cox (Risques Proportionnels de Cox) est une régression semi-paramétrique qui analyse l'impact de chaque variable sur le risque de décès.

    - L’ajout d’expertise humaine permet d’améliorer le modèle en intégrant des connaissances médicales sous forme de nouvelles variables. Par exemple, on pourra effectuer des ratios entre certaines variables indiquant si le risque est plus élevé ou plus faible (c'est de l'enrichissement de données).

- LightGBM + modèle de Cox + expertise médicale (LightGBM a priori moins robuste que XGBoost pour notre quantité de données).

    - LightGBM (Light Gradient Boosting Machine) est un algorithme de machine learning basé sur le Gradient Boosting, optimisé pour être rapide et efficace sur de grands volumes de données tabulaires. Contrairement aux méthodes classiques qui construisent les arbres de décision niveau par niveau (depth-wise), LightGBM adopte une approche leaf-wise, où il développe en priorité les feuilles d’arbre ayant le plus grand gain d'information, ce qui accélère l'entraînement et améliore la précision sur de grands datasets.

    - Le modèle de Cox (Risques Proportionnels de Cox) est une régression semi-paramétrique qui analyse l'impact de chaque variable sur le risque de décès.

    - L’ajout d’expertise humaine permet d’améliorer le modèle en intégrant des connaissances médicales sous forme de nouvelles variables. Par exemple, on pourra effectuer des ratios entre certaines variables indiquant si le risque est plus élevé ou plus faible (c'est de l'enrichissement de données).
#fghgfgh