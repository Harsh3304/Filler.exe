import threading
import numpy as np
import torch
from tkinter import Tk, Label, Button, PhotoImage, filedialog
from torchvision import transforms
from PIL import Image
import os
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_SIZE = 256  
ENLARGED_SIZE = 200  
file_path = os.path.dirname(os.path.abspath(__file__))  

def load_and_process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image) / 255.0                                       

def create_noisy_image(true_image_np):
    noise = np.random.normal(0, 0.1, true_image_np.shape)                     
    noisy_image = np.clip(true_image_np + noise, 0, 1)  
    return noisy_image

def create_gap_based_noise(image_np, gap_size=20, num_gaps=5):
    noisy_image = image_np.copy()
    height, width, _ = noisy_image.shape

    for _ in range(num_gaps):
        x_start = np.random.randint(0, width - gap_size)                        
        y_start = np.random.randint(0, height - gap_size)

        noisy_image[y_start:y_start + gap_size, x_start:x_start + gap_size] = 0    

    return noisy_image

def np_to_torch(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

def np_to_image(np_array, max_size=(ENLARGED_SIZE, ENLARGED_SIZE)):
    image = Image.fromarray((np_array * 255).astype(np.uint8))                     
    image.thumbnail(max_size, Image.LANCZOS)                                        

    bio = BytesIO()
    image.save(bio, format="PNG")
    bio.seek(0)
    return PhotoImage(data=bio.getvalue()), image                                   

class CNN_configurable(torch.nn.Module):
    def __init__(self, n_lay=3, n_chan=64, ksize=3):
        super(CNN_configurable, self).__init__()
        layers = []
        in_channels = 3  

        for _ in range(n_lay):
            layers.append(torch.nn.Conv2d(in_channels, n_chan, ksize, padding=ksize // 2))
            layers.append(torch.nn.ReLU())
            in_channels = n_chan

        layers.append(torch.nn.Conv2d(n_chan, 3, ksize, padding=ksize // 2))  # Output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model():
    global true_object_np, final_image_label, epoch_label, loss_label, best_image_pil 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    true_object_torch = np_to_torch(true_object_np.transpose(2, 0, 1)).to(device)

    input_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # NRMSE loss function
    def nrmse_fn(recon, reference):
        n = (reference - recon) ** 2 
        den = reference ** 2
        return 100.0 * torch.mean(n) ** 0.5 / torch.mean(den) ** 0.5

    cnn = CNN_configurable(n_lay=3, n_chan=64, ksize=3).to(device)
    optimiser = torch.optim.Adam(cnn.parameters(), lr=1e-4)

    
    best_loss = float('inf')
    best_image = None
    train_loss = []
    nrmse_list = []

    # Training loop
    for ep in range(100000):  # Adjust 
        optimiser.zero_grad()
        output_image = cnn(input_image)
        loss = nrmse_fn(output_image, true_object_torch.unsqueeze(0))
        loss.backward()
        optimiser.step()

        train_loss.append(loss.item())
        nrmse = nrmse_fn(output_image, true_object_torch.unsqueeze(0))
        nrmse_list.append(nrmse.item())

        epoch_label.config(text=f"{ep}")
        loss_label.config(text=f"{loss.item():.4f}")

        if ep % 1000 == 0:
            print(f'Epoch {ep}, Loss: {loss.item()}')
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_image = output_image.detach().cpu().numpy()
                best_image_pil = np_to_image(best_image[0].transpose(1, 2, 0), max_size=(ENLARGED_SIZE, ENLARGED_SIZE))[1]

            if ep > 0:
                ax_loss.clear()
                ax_loss.plot(nrmse_list, label='NRMSE')
                ax_loss.set_title('NRMSE')
                ax_loss.legend()
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('NRMSE')
                canvas.draw()

    if best_image is not None:
        best_image_np = np.clip(best_image[0].transpose(1, 2, 0), 0, 1)
        best_image_preview, best_image_pil = np_to_image(best_image_np)

        final_image_label.config(image=best_image_preview)
        final_image_label.image = best_image_preview  

def start_training():
    training_thread = threading.Thread(target=train_model)
    training_thread.start()

def add_image():
    global true_object_np, true_image_label, data_image_label

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )

    if file_path:  
        true_object_np = load_and_process_image(file_path)  
        noisy_image = create_noisy_image(true_object_np)
        gap_noisy_image = create_gap_based_noise(noisy_image) 
        true_image = np_to_image(true_object_np, max_size=(ENLARGED_SIZE, ENLARGED_SIZE))
        data_image = np_to_image(gap_noisy_image, max_size=(ENLARGED_SIZE, ENLARGED_SIZE))

        true_image_label.config(image=true_image[0])
        true_image_label.image = true_image[0]  

        data_image_label.config(image=data_image[0])
        data_image_label.image = data_image[0]  

def download_best_image():
    global best_image_pil

    if best_image_pil is not None:
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if file_path:
            best_image_pil.save(file_path)
            print("Image saved:", file_path)
    else:
        print("No best epoch image available to download.")

def main():
    global BG_Label, Add_image_bt_img, data_image_label, true_image_label, final_image_label, epoch_label, loss_label, ax_loss, canvas, best_image_pil  # Include new labels here

    best_image_pil = None  # Initialize best_image_pil to store the PIL image of the best epoch

    window = Tk()
    window.geometry('1199x767')
    window.maxsize(1199, 767)
    window.minsize(1199, 767)
    window.title("Gap Filling - ProtoType")
    window.configure(background='white')

    BG = PhotoImage(file=file_path + "/Dependencies/BG1.png") 
    Add_image_bt_img = PhotoImage(file=file_path + "/Dependencies/Photo1.png") 
    Denoise_bt_img = PhotoImage(file=file_path + "/Dependencies/denoise1.png") 
    Superimpose_bt_img = PhotoImage(file=file_path + "/Dependencies/superimpose1.png") 
    Colourize_bt_img = PhotoImage(file=file_path + "/Dependencies/colourize1.png") 
    Download_bt_img = PhotoImage(file=file_path + "/Dependencies/download1.png") 
    

    BG_Label = Label(window, image=BG, border=0)
    BG_Label.place(x=0, y=0)

    true_image_label = Label(window, bg="#2C2C2C")
    true_image_label.place(x=405, y=125)  

    data_image_label = Label(window, bg="#2C2C2C")
    data_image_label.place(x=685, y=125) 

    final_image_label = Label(window, bg="#2C2C2C")
    final_image_label.place(x=952, y=125)  

    epoch_label = Label(window, text="0", bg="#2C2C2C", fg="white", font=("Arial Black", 17))
    epoch_label.place(x=960, y=400)

    loss_label = Label(window, text="N/A", bg="#2C2C2C",fg="white", font=("Arial Black", 17))
    loss_label.place(x=960, y=470)

    denoise_bt = Button(window, image= Denoise_bt_img, border=0, bg="#2C2C2C",activebackground="#2C2C2C", command=start_training)
    denoise_bt.place(x=11, y=150)

    superimpose_bt = Button(window, image= Superimpose_bt_img, border=0, bg="#2C2C2C", activebackground="#2C2C2C", command="")
    superimpose_bt.place(x=11, y=210)

    colourize_bt = Button(window, image= Colourize_bt_img, border=0, bg="#2C2C2C", activebackground="#2C2C2C", command="")
    colourize_bt.place(x=11, y=270)

    add_image_bt = Button(window, image= Add_image_bt_img, border=0, bg="#161616", activebackground="#161616", command=add_image)
    add_image_bt.place(x=1110, y=10)

    download_bt = Button(window, image= Download_bt_img, border=0,  bg="#161616", activebackground="#161616", command=download_best_image)
    download_bt.place(x=934, y=540)


    fig, ax_loss = plt.subplots(figsize=(5, 4), dpi=95)
    fig.set_facecolor('#2C2C2C') 
    ax_loss.set_facecolor('#2C2C2C')  
    
    ax_loss.set_title('NRMSE', color='white')  
    ax_loss.set_xlabel('Epoch', color='white')  
    ax_loss.set_ylabel('NRMSE', color='white')  

    ax_loss.tick_params(axis='x', colors='white')  
    ax_loss.tick_params(axis='y', colors='white') 


    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().place(x=410, y=365)

    window.mainloop()

if __name__ == "__main__":
    main()
