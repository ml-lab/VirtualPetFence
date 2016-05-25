# VirtualPetFence

## Disclamer
Det meste av koden for å konvertere fra caffemodell til tensorflow modell er tatt fra: https://github.com/ry/tensorflow-resnet.
Dette er filene "convert.py", "resnet.py", "forward.py" og "synset.py".

## Avhengigheter
Programmet er avhengig av **Tensorflow master**, altså man må bygge nyeste fra github, ikke versjon 0.80.
I tillegg kommer avhengighetene numpy og OpenCV

For å få treningen av nettverket til å fungere må "pathene" til treningsdata være riktig. Get dataset tar inn

Fortrente modell kan lastes ned på: https://drive.google.com/open?id=0B8EJ9UWrW_JRQ0RZWEJIMGVRTUU

For å trene egene modeller må man ha en fortrent ResNet-ImageNet modell:
En ferdig konvertert (til tensorflow) modell kan lastes ned fra https://github.com/ry/tensorflow-resnet (ikke sikker på at dette fungerer)
Eller du kan konvertere caffe modellen selv ved å laste ned den fra https://github.com/KaimingHe/deep-residual-networks og så kjøre convert.py

## Her er oversikten over de viktige filene:

*convert.py* converterer og lager en tensorflow modell som jeg bruker som utgangspunkt

*train_catnet.py* Her trener man modellen

*runSegmentation.py* Dette er filen en bruker kjører. CatFinder må initisaliseres med stien til der model filene ligger
 (checkpoint, model og model-meta). Den laster inn modellen (tar tid), og man får et GUI for å tegne
og se hva som foregår. Den spiller også av en lyd, hvis den finner en katt i området.
