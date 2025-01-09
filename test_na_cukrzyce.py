import pandas as pd
import os
from fpdf import FPDF

os.makedirs("raporty", exist_ok=True)


# Ładowanie modelu i przetwarzacza (pretrained pipeline)
def load_model_and_preprocessor():
    import joblib
    model = joblib.load("lightgbm_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


def get_input(prompt, valid_type, condition=lambda x: True, error_message="Nieprawidłowa wartość, spróbuj ponownie."):
    while True:
        try:
            value = valid_type(input(prompt))
            if condition(value):
                return value
            else:
                print(error_message)
        except ValueError:
            print(error_message)


def predict_user_data():
    print("Wprowadź swoje dane w celu oszacowania prawdopodobieństwa cukrzycy:")
    user_data = {}

    # Obliczenie BMI
    weight = get_input("Podaj swoją wagę w kilogramach: ", float, lambda x: x > 0, "Waga musi być liczbą większą od 0.")
    height = get_input("Podaj swój wzrost w metrach (np. 1.75): ", float, lambda x: x > 0, "Wzrost musi być liczbą większą od 0.")
    user_data["BMI"] = weight / (height ** 2)

    # Kategoria wiekowa
    age = get_input("Podaj swój wiek: ", int, lambda x: x > 0, "Wiek musi być liczbą większą od 0.")
    if age < 25:
        user_data["Age"] = 1
    elif age < 35:
        user_data["Age"] = 2
    elif age < 45:
        user_data["Age"] = 3
    elif age < 55:
        user_data["Age"] = 4
    elif age < 65:
        user_data["Age"] = 5
    else:
        user_data["Age"] = 6

    # Dochód
    income = get_input("Podaj swój dochód miesięczny netto w zł: ", float, lambda x: x >= 0, "Dochód musi być liczbą większą lub równą 0.")
    if income < 2000:
        user_data["Income"] = 1
    elif income < 4000:
        user_data["Income"] = 2
    elif income < 6000:
        user_data["Income"] = 3
    elif income < 8000:
        user_data["Income"] = 4
    elif income < 10000:
        user_data["Income"] = 5
    elif income < 15000:
        user_data["Income"] = 6
    elif income < 20000:
        user_data["Income"] = 7
    else:
        user_data["Income"] = 8

    # Wykształcenie
    user_data["Education"] = get_input(
        "Podaj poziom wykształcenia (1: brak szkoły lub przedszkole, 2: szkoła podstawowa, 3: gimnazjum, 4: liceum lub technikum, 5: studia rok 1-5, 6: ukończone studia wyższe): ",
        int,
        lambda x: x in [1, 2, 3, 4, 5, 6],
        "Wybierz wartość od 1 do 6.")

    # Pozostałe zmienne
    user_data["PhysHlth"] = get_input("Ile dni w ciągu ostatnich 30 dni miałeś problemy zdrowotne fizyczne (0-30): ", int, lambda x: 0 <= x <= 30)
    user_data["MentHlth"] = get_input("Ile dni w ciągu ostatnich 30 dni miałeś problemy zdrowotne psychiczne (0-30): ", int, lambda x: 0 <= x <= 30)
    user_data["GenHlth"] = get_input("Jak ocenisz swoje zdrowie ogólne (1: doskonałe, 2: bardzo dobre, 3: dobre, 4: dostateczne, 5: słabe): ", int, lambda x: x in [1, 2, 3, 4, 5])
    user_data["HighBP"] = get_input("Czy masz nadciśnienie tętnicze? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["HighChol"] = get_input("Czy masz wysoki poziom cholesterolu? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["CholCheck"] = get_input("Czy przeprowadziłeś badanie cholesterolu w ciągu ostatnich 5 lat? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["Smoker"] = get_input("Czy wypaliłeś co najmniej 100 papierosów w całym życiu? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["Stroke"] = get_input("Czy kiedykolwiek miałeś udar? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["HeartDiseaseorAttack"] = get_input("Czy kiedykolwiek miałeś chorobę wieńcową lub zawał serca? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["PhysActivity"] = get_input("Czy wykonywałeś aktywność fizyczną w ciągu ostatnich 30 dni? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["Fruits"] = get_input("Czy spożywasz owoce co najmniej raz dziennie? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["Veggies"] = get_input("Czy spożywasz warzywa co najmniej raz dziennie? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["HvyAlcoholConsump"] = get_input("Czy spożywasz nadmierne ilości alkoholu? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["AnyHealthcare"] = get_input("Czy masz jakąkolwiek opiekę zdrowotną? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["NoDocbcCost"] = get_input("Czy w ciągu ostatnich 12 miesięcy nie mogłeś odwiedzić lekarza z powodu kosztów? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["DiffWalk"] = get_input("Czy masz trudności z chodzeniem lub wchodzeniem po schodach? (0: nie, 1: tak): ", int, lambda x: x in [0, 1])
    user_data["Sex"] = get_input("Podaj swoją płeć (0: kobieta, 1: mężczyzna): ", int, lambda x: x in [0, 1])

    user_df = pd.DataFrame([user_data])
    model, preprocessor = load_model_and_preprocessor()
    user_scaled = preprocessor.transform(user_df)
    prediction_proba = model.predict_proba(user_scaled)[:, 1][0]
    print(f"Prawdopodobieństwo cukrzycy: {prediction_proba * 100:.2f}%")
    generate_user_report(prediction_proba)


# Generowanie raportu dla użytkownika
def generate_user_report(probability):
    pdf = PDF()
    pdf.add_page()
    pdf.add_section("Wynik Predykcji", f"Prawdopodobienstwo cukrzycy: {probability * 100:.2f}%")
    if probability > 0.75:
        pdf.add_section("Zalecenie", "Model sugeruje wysokie ryzyko cukrzycy. Skonsultuj się z lekarzem.")
    else:
        pdf.add_section("Zalecenie", "Model sugeruje niskie ryzyko cukrzycy. Kontynuuj zdrowy tryb zycia.")

    pdf.output("raporty/Raport_Predykcji_Uzytkownika.pdf")
    print("Raport predykcji zapisano jako 'Raport_Predykcji_Uzytkownika.pdf'.")


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Raport Predykcji Cukrzycy', align='C', ln=1)
        self.ln(10)

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=1)
        self.ln(5)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)
        self.ln()


if __name__ == "__main__":
    predict_user_data()
