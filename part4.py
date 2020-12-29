import pickle

from run_attention import find_tfl_lights
from tensorflow.keras.models import load_model
from building_data_set import crop
import matplotlib.pyplot as plt
from SFM import calc_TFL_dist
import numpy as np
import SFM


def visualize(image, candidates, colors, fig, title):
    fig.set_title(title)
    fig.imshow(image)
    fig.set_xticks([])
    fig.set_yticks([])

    x_red = [x[1] for i, x in enumerate(candidates) if colors[i] == 'red']
    y_red = [y[0] for i, y in enumerate(candidates) if colors[i] == 'red']

    x_green = [x[1] for i, x in enumerate(candidates) if colors[i] == 'green']
    y_green = [y[0] for i, y in enumerate(candidates) if colors[i] == 'green']

    fig.plot(x_red, y_red, 'r.')
    fig.plot(x_green, y_green, 'g.')


def visualize3(img, prev_container, curr_container, focal, pp, EM, fig):
    if not len(curr_container.tfl):
        return

    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp, EM)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))

    fig.set_title('Tfl Distances')
    fig.imshow(img)
    curr_p = np.array(curr_container.tfl)
    fig.plot(curr_p[:, 1], curr_p[:, 0], 'b+')

    for i in range(len(curr_p)):
        fig.plot([curr_p[i, 1], foe[0]], [curr_p[i, 0], foe[1]], 'b')
        if curr_container.valid[i]:
            fig.text(curr_p[i, 1], curr_p[i, 0], r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]),
                     color='r', fontsize=6)

    fig.plot(foe[0], foe[1], 'r+')
    fig.plot(rot_pts[:, 1], rot_pts[:, 0], 'g+')

    fig.set_xticks([])
    fig.set_yticks([])


class ProcessData:
    def __init__(self, pls_path):
        data = []
        with open(pls_path, 'r') as play_list:
            for line in play_list:
                data.append(line[:-1])
        self.pls_data = data
        print("process")

    def get_frames(self):
        return self.pls_data[1:]

    def read_pkl_file(self):
        with open(self.pls_data[0], 'rb') as pkl_file:
            data = pickle.load(pkl_file, encoding='latin1')
        return data['principle_point'], data['flx'], data


class FrameContainer:
    def __init__(self, tfl, auxiliary):
        self.auxiliary = auxiliary
        self.tfl = tfl
        self.traffic_lights_3d_location = []
        self.valid = []


class Controller:
    def __init__(self, pls_path):
        self.process_data = ProcessData(pls_path)
        pp, focal, self.EMs_data = self.process_data.read_pkl_file()
        self.tfl_manager = TFLManager(pp, focal)
        self.frames = self.process_data.get_frames()

    def run(self):
        for index, img in enumerate(self.frames):
            if index != 0:
                self.tfl_manager.run(img, self.EMs_data['egomotion_' + str(index - 1 + 24) + '-' + str(index + 24)])
            else:
                self.tfl_manager.run(img)


class TFLManager:
    def __init__(self, pp, focal):
        self.pp = pp
        self.focal = focal
        self.prev = None

    def detect_lights_adapter(self, img_path, fig):
        y_red, x_red, y_green, x_green = find_tfl_lights(img_path)
        assert len(x_red) == len(y_red) and len(x_green) == len(y_green)

        candidates = [[x, y] for x, y in zip(x_red, y_red)]
        candidates += [[x, y] for x, y in zip(x_green, y_green)]
        auxiliary = ['red' for i in range(len(y_red))]
        auxiliary += ['green' for i in range(len(y_green))]

        visualize(plt.imread(img_path).copy(), candidates, auxiliary, fig, "Light Detection")

        return np.array(candidates), np.array(auxiliary)

    def verify_tfl_adapter(self, candidates, auxiliary, img_path, fig):
        img = plt.imread(img_path) * 255
        img = img.astype('uint8')
        print("****loading model****")
        loaded_model = load_model("model.h5")
        print("****loaded model****")
        croped_images = [crop(candidate[0], candidate[1], img) for candidate in candidates]

        candidates = [can for i, can in enumerate(candidates) if croped_images[i].shape == (81, 81, 3)]
        auxiliary = [aux for i, aux in enumerate(auxiliary) if croped_images[i].shape == (81, 81, 3)]
        croped_images = np.array([x for x in croped_images if x.shape == (81, 81, 3)])

        croped_images *= 255

        prediction = loaded_model.predict(np.array(croped_images))

        candidates = [candidate for index, candidate in enumerate(candidates) if prediction[index, 1] > 0.9995]
        auxiliary = [aux for index, aux in enumerate(auxiliary) if prediction[index, 1] > 0.9995]

        visualize(img, candidates, auxiliary, fig, "TFL Detection")

        return candidates, auxiliary

    def calc_dist_adapter(self, curr, EM):
        return calc_TFL_dist(self.prev, curr, self.focal, self.pp, EM)

    def run(self, img_path, EM=None):
        fig, (light_src, tfl, distances) = plt.subplots(3, 1, figsize=(20, 10))

        candidates, auxiliary = self.detect_lights_adapter(img_path, light_src)
        tfl, auxiliary = self.verify_tfl_adapter(candidates, auxiliary, img_path, tfl)
        curr = FrameContainer(tfl, auxiliary)

        if self.prev is not None:
            curr = self.calc_dist_adapter(curr, EM)
            visualize3(plt.imread(img_path), self.prev, curr, self.focal, self.pp, EM, distances)

        self.prev = curr

        plt.show()


def main():
    controller = Controller(r"C:\Users\chevi\Documents\bootcamp\traffic_light\the_project\New folder\data.pls")
    controller.run()


if __name__ == "__main__":
    main()
