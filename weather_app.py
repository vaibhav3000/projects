import tkinter as tk
from tkinter import ttk
import requests
from tkinter import messagebox
from PIL import Image, ImageTk

def get_weather(city):
    API_key = "ee66b181dbc72c4deff96f71f69be7ac"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}"
    res = requests.get(url)

    if res.status_code == 404:
        messagebox.showerror("Error", "City Not Found")
        return None

    weather = res.json()
    icon_id = weather["weather"][0]["icon"]
    temperature = weather["main"]["temp"] - 273.15
    description = weather["weather"][0]["description"]
    city_name = weather["name"]
    country = weather["sys"]["country"]

    icon_url = f"https://openweathermap.org/img/wn/{icon_id}@2x.png"
    return (icon_url, temperature, description, city_name, country)

def search():
    city = city_entry.get()
    result = get_weather(city)
    if result is None:
        return
    
    icon_url, temperature, description, city_name, country = result

    location_label.configure(text=f"{city_name}, {country}")

    image = Image.open(requests.get(icon_url, stream=True).raw)
    icon = ImageTk.PhotoImage(image)
    icon_label.configure(image=icon)
    icon_label.image = icon

    temperature_label.configure(text=f"{temperature:.2f}°C")
    description_label.configure(text=f"Description: {description}")

root = tk.Tk()
root.title("Weather App")
root.geometry("500x500")

city_entry = tk.Entry(root, font=("Helvetica", 18, "bold"))
city_entry.pack(pady=10)

search_button = tk.Button(root, text="Search Weather", command=search, font=("Helvetica", 14))
search_button.pack(pady=10)

location_label = tk.Label(root, font=("Helvetica", 20))
location_label.pack(pady=20)

icon_label = tk.Label(root)
icon_label.pack()

temperature_label = tk.Label(root, font=("Helvetica", 20))
temperature_label.pack()    

description_label = tk.Label(root, font=("Helvetica", 20))
description_label.pack()

root.mainloop()