\# Zadanie: symulacja i sterowanie quadrocopterem (portfolio, Automatyka i Robotyka)



Zbuduj od zera projekt w Pythonie 3.11+. Liczy się poprawność fizyczna i matematyczna,

czytelny kod, testy i efektowne wyniki. Wszystko po angielsku w kodzie, README po polsku.



\## 1. Model (src/quadrotor/dynamics.py, params.py)

\- Ciało sztywne 6-DOF: pozycja i prędkość w układzie inercjalnym, orientacja jako kwaternion,

&#x20; prędkość kątowa w układzie ciała. Równania Newtona-Eulera z momentem żyroskopowym.

\- Konfiguracja X, cztery wirniki: ciąg T\_i = kf \* w\_i^2, moment reakcyjny = km \* w\_i^2,

&#x20; dynamika silnika jako człon pierwszego rzędu, saturacja prędkości wirników.

\- Mikser: (ciąg całkowity, momenty) -> prędkości wirników i odwrotnie.

\- Opór aerodynamiczny liniowy, model wiatru (stały + porywy Dryden lub OU).

\- Wszystkie parametry (masa, tensor bezwładności, ramię, kf, km, stała czasowa silnika)

&#x20; w jednym pliku konfiguracyjnym YAML lub dataclass.

\- Integracja RK4 ze stałym krokiem (fizyka 1 kHz, sterowanie 250 Hz, EKF 100 Hz).

&#x20; Renormalizacja kwaternionu.



\## 2. Regulatory (src/quadrotor/control/)

\- Baseline: kaskadowy PID (pozycja -> prędkość -> kąty -> prędkości kątowe), anti-windup.

\- Regulator geometryczny na SO(3) (Lee, Leok, McClamroch 2010), śledzenie pozycji i yaw.

\- LQR do zawisu na modelu zlinearyzowanym, z uzasadnionym doborem Q i R.

\- Opcjonalnie liniowe MPC (cvxpy) do śledzenia trajektorii z ograniczeniami na ciąg i kąty.

&#x20; Jeśli za wolne, ogranicz do horyzontu 20 kroków.



\## 3. Trajektorie (src/quadrotor/planning/)

\- Generator trajektorii min-snap przez punkty pośrednie (Mellinger \& Kumar), wielomiany 7. rzędu,

&#x20; rozwiązanie jako QP z ciągłością pochodnych.

\- Gotowe scenariusze: ósemka 3D, helisa, slalom przez bramki, agresywny zwrot.

\- Feed-forward prędkości, przyspieszenia i jerku do regulatora.



\## 4. Estymacja (src/quadrotor/estimation/)

\- Symulowane sensory: IMU (szum + bias random walk), GPS 10 Hz z szumem, barometr.

\- EKF (lub ESKF) estymujący pozycję, prędkość, orientację i biasy IMU.

\- Regulatory pracują na estymacie. Wykres błędu estymacji i 3-sigma.



\## 5. Eksperymenty (scripts/)

Porównaj PID vs geometryczny vs LQR/MPC na tych samych scenariuszach:

\- RMSE śledzenia pozycji, maksymalny błąd, zużycie energii, czas do stabilizacji.

\- Odporność: wiatr, błąd masy +-20%, awaria (spadek sprawności jednego wirnika o 30%).

\- Zapisz wykresy (matplotlib) do results/ oraz tabelę metryk do results/metrics.csv i README.

\- Animacja 3D (matplotlib lub plotly) zapisana jako GIF w results/, min. dla ósemki i slalomu.



\## 6. Wymagania techniczne

\- Struktura pakietu src/quadrotor/, scripts/, tests/, pyproject.toml lub requirements.txt.

\- Type hints, docstringi, ruff lub flake8 bez błędów.

\- Testy pytest: zachowanie energii bez tarcia i sterowania, norma kwaternionu, zawis w równowadze

&#x20; (ciąg = mg), zbieżność regulatora geometrycznego, spełnianie ciągłości w min-snap,

&#x20; zgodność linearyzacji z modelem nieliniowym dla małych odchyłek.

\- GitHub Actions: workflow uruchamiający testy i lint.

\- Pełny zestaw eksperymentów ma działać na CPU w kilka minut: python scripts/run\_all.py



\## 7. README (po polsku)

Opis problemu, kluczowe równania, opis każdego regulatora, tabela wyników, wykresy, GIF-y,

instrukcja uruchomienia jednym poleceniem, sekcja "Co dalej" (np. SLAM, RL, model 3D w Gazebo).



\## Sposób pracy

Pracuj etapami i commituj po każdym działającym etapie:

model -> testy modelu -> PID -> geometryczny -> trajektorie -> EKF -> LQR/MPC -> eksperymenty -> CI -> README.

Na końcu uruchom testy i wszystkie skrypty, upewnij się, że wykresy i GIF-y są w repo,

i otwórz pull request z podsumowaniem: co działa, wyniki porównania, znane ograniczenia.



