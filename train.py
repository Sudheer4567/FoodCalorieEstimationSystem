import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Data Preprocessing ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data_dir = 'dataset'
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Adjust based on your task
)

# --- Model Definition (Transfer Learning with ResNet50) ---
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x)  # 101 output classes
model = Model(inputs=base_model.input, outputs=predictions)

# --- Compile the Model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train the Model ---
model.fit(
    train_generator,
    epochs=10,  # Adjust as needed
)

# --- Save the Model ---
model.save('models/food_model.keras')