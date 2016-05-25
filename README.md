# VirtualPetFence

Det meste av koden for å konvertere fra caffemodell til tensorflow modell er tatt fra: https://github.com/ry/tensorflow-resnet.
Å få treningen til å kjøre fra github er nok ikke helt enkelt siden alle pathene er hardkodet og litt rundt om kring i koden.


Her er oversikten over de viktige filene:

*convert.py* converterer og lager en tensorflow modell som jeg bruker som utgangspunkt

*train_catnet.py* Her trener man modellen

*runSegmentation.py* Dette er filen en bruker kjører. Den laster inn modellen (tar tid), og man får et GUI for å tegne
og se hva som foregår. Den spiller også av en lyd, hvis den finner en katt i området.