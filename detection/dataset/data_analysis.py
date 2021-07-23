import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from tqdm import tqdm

from detection.dataset.clustering import avg_iou, kmeans_iou
from detection.dataset.dataset_generator import Dataset


class Analyser:
    def __init__(self, model_input_size, dataset):

        self.img_size = dataset.img_width, dataset.img_height
        self.input_size = model_input_size
        self.labels = np.concatenate(dataset.labels).astype(dtype=np.float64)

        self.rescale_labels()

    def rescale_labels(self):
        # scale the predicted boxes to the size of the models input images
        self.labels[..., [0, 2]] = self.labels[..., [0, 2]] * (self.input_size[0] / self.img_size[0])
        self.labels[..., [1, 3]] = self.labels[..., [1, 3]] * (self.input_size[1] / self.img_size[1])

    def to_relative(self):
        self.labels[..., [0, 2]] /= self.img_size[0]
        self.labels[..., [1, 3]] /= self.img_size[1]

    def get_sides(self):
        """

        :return: Shape: (#boxes, 2) with last axis: (width, height)
        """
        side_len = self.labels[..., [2, 3]] - self.labels[..., [0, 1]]
        return side_len

    def get_ratios(self):
        """

        :return: Shape: (#boxes, )
        """
        sides = self.get_sides()
        return sides[..., 0] / sides[..., 1]

    def get_centers(self):
        return (self.labels[..., [0, 1]] + self.labels[..., [2, 3]]) / 2

    def ratio_histo(self):
        # ration = width / height
        ratios = self.get_ratios()
        plt.hist(ratios, bins=80, alpha=0.5)

        mean = np.mean(ratios)
        std = np.std(ratios)
        print("Mean: {0:.2f} Std: {1:.2f}".format(mean, std))

        plt.axvline(mean, color='r', linestyle='--')
        plt.axvline(mean + std, color='g', linestyle='--')
        plt.axvline(mean - std, color='g', linestyle='--')
        plt.xlabel('Seitenverhältnis (Breite / Höhe)')
        plt.ylabel('Anzahl Objekte')
        plt.xlim(0.0, 2.0)
        plt.show()

    def ratio_hist2d(self):
        sides = self.get_sides()
        min_w, max_w = np.min(sides[..., 0]), np.max(sides[..., 0])
        min_h, max_h = np.min(sides[..., 1]), np.max(sides[..., 1])
        hist = plt.hist2d(x=sides[..., 0], y=sides[..., 1], bins=min(max_w - min_w, max_h - min_h), cmap='cividis')
        plt.xticks(hist[1], range(min_w, max_w + 1, 1))
        plt.yticks(hist[2], range(min_h, max_h + 1, 1))
        plt.xlabel('width')
        plt.ylabel('height')
        cb = plt.colorbar(hist[3])
        cb.set_label('count')
        plt.show()

    def scatter_ratio(self, centers, labels=None, name='kMeans', show=True, save=True):
        sides = self.get_sides()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(sides[:, 0], sides[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5)
        ax.scatter(centers[:, 0], centers[:, 1], marker='x', s=180, linewidths=4,
                   color='w', zorder=10)
        ax.set_xlabel('Boxbreite [px]')
        ax.set_ylabel('Boxhöhe [px]')
        ax.set_xlim((0, np.max(sides)))
        ax.set_ylim((0, np.max(sides)))
        if show:
            plt.show()
        if save:
            fig.savefig(name)

    def ratio_euclid_means(self, c=3, show=False, verbose=True, save=False, tries=10):
        sides = self.get_sides()
        best = [0.0, None, None]

        for _ in tqdm(range(tries), desc='Groups: {0}'.format(c)):
            kmeans = KMeans(n_clusters=c, verbose=0, random_state=None)
            res = kmeans.fit(sides)
            centers = res.cluster_centers_
            labels = res.labels_
            accuracy = avg_iou(sides, centers)
            if accuracy > best[0]:
                best = [accuracy, centers, labels]
        centers, labels = best[1], best[2]
        ratios = np.around(centers[:, 0] / centers[:, 1], decimals=3).tolist()
        ratios = sorted(ratios)

        if verbose:
            ratios = centers[:, 0] / centers[:, 1]
            print("Accuracy: {:.2f}%".format(avg_iou(sides, centers) * 100))
            print("Boxes:\n {}".format(centers))
            print("Ratios:\n {}".format([round(i, 2) for i in sorted(ratios)]))
        if show or save:
            self.scatter_ratio(centers, labels=labels, show=show, save=save,
                               name='kMeans_{0}_{1:.3f}.png'.format(c, best[0]))

        return {'groups': c, 'dist': 'euclid', 'accuracy': best[0], 'rations': ratios}

    def ratio_iou_means(self, c=5, show=False, save=False, verbose=True, dist=np.median, tries=10):
        sides = self.get_sides()
        best = [0.0, None, None]
        for _ in tqdm(range(tries), desc='Groups: {0}'.format(c)):
            centers, labels = kmeans_iou(sides, k=c, dist=dist)
            accuracy = avg_iou(sides, centers)
            if accuracy > best[0]:
                best = [accuracy, centers, labels]
        centers, labels = best[1], best[2]
        ratios = np.around(centers[:, 0] / centers[:, 1], decimals=3).tolist()
        ratios = sorted(ratios)

        str_dist = 'mean' if dist == np.mean else 'median'
        if verbose:
            print("Accuracy: {:.2f}%".format(best[0] * 100))
            print("Boxes:\n {}".format(centers))
            print("Boxes norm:\n {}".format(centers / 240))
            print("Ratios:\n {}".format(ratios))
        if show or save:
            self.scatter_ratio(centers, labels=labels, show=show, save=save,
                               name='iou_kMeans_{0}_{1:.3f}_{2}.png'.format(c, best[0], str_dist))
        return {'groups': c, 'dist': str_dist, 'accuracy': best[0], 'rations': ratios}

    def test_kmeans_iou(self, cmin, cmax, tries=10, dist=np.median, save=True):
        collection = []
        for c in tqdm(range(cmin, cmax + 1), desc='Trying different group sizes.'):
            result = self.ratio_iou_means(c=c, show=False, save=save, verbose=False, dist=dist, tries=tries)
            collection.append(result)
        return collection

    def test_kmeans_eucl(self, cmin, cmax, tries=10, save=True):
        collection = []
        for c in tqdm(range(cmin, cmax + 1), desc='Trying different group sizes.'):
            result = self.ratio_euclid_means(c=c, show=False, save=save, verbose=False, tries=tries)
            collection.append(result)
        return collection

    def ratio_join(self):
        sides = self.get_sides().astype(np.float)
        a = sns.jointplot(sides[:, 0], sides[:, 1])
        plt.show()

    def scales(self):
        scale_relative_side = min(self.input_size)
        sides = self.get_sides()
        sns.set_theme(style="whitegrid")
        print(sides.shape)
        scale = sides / scale_relative_side
        df_x = pd.DataFrame(scale[:, 0], columns=["value"])
        df_x["axis"] = "x"
        df_y = pd.DataFrame(scale[:, 1], columns=["value"])
        df_y["axis"] = "y"
        df = pd.concat([df_x, df_y])
        df.to_csv("scale_data.csv", index=False)
        sns.displot(data=df, x="value", hue="axis", palette='dark', alpha=0.8, kind="hist",  stat='density')
        plt.xlabel('Relative Boxgröße')
        plt.ylabel('Dichte')
        plt.tight_layout()
        # plt.gcf().savefig('scale_histogram.png')
        plt.show()

    def box_distribution(self, mode='scatter'):
        centers = self.get_centers()

        if mode == 'scatter':
            plt.scatter(centers[:, 0], centers[:, 1], s=5, alpha=0.4)
        elif mode == 'heatmap':
            bins = 30
            heatmap, xedges, yedges = np.histogram2d(centers[:, 0], centers[:, 1],
                                                     bins=(self.img_size[0] // bins, self.img_size[1] // bins))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(heatmap.T, extent=extent, origin='upper')
        elif mode == "join":
            jn = sns.jointplot(centers[:, 0], centers[:, 1], ratio=5, space=0.1, kind="scatter",
                               xlim=(-10, self.input_size[0]), ylim=(-10, self.input_size[1]),
                               joint_kws={'alpha': 0.1})
            jn.fig.set_figwidth(8)
            jn.fig.set_figheight(5)
            jn.ax_joint.set_xlabel("Width")
            jn.ax_joint.set_ylabel("Height")
        plt.gca().invert_yaxis()
        plt.show()

    def scale_values(self, clusters=10):
        sides = np.array(list(map(lambda label: self.get_side_len(label, relative=True), self.labels)))
        sides_reshaped = sides.flatten().reshape((-1, 1))
        k_means = KMeans(n_clusters=clusters, random_state=0, verbose=0).fit(sides_reshaped)
        y_means = k_means.predict(sides_reshaped)
        centers = k_means.cluster_centers_.reshape(clusters)
        v, y = np.unique(y_means, return_counts=True)
        sorted_centers = np.round(np.sort(centers), decimals=3)
        sorted_y = y[centers.argsort()]
        # heatmap([sorted_y], xticklabels=sorted_centers, square=True, cbar_kws={"orientation": "horizontal"})
        plt.plot(sorted_centers, sorted_y)
        plt.xlabel("Bounding box scales")
        plt.title("Bbox cluster heatmap")
        plt.yticks([])
        plt.show()


if __name__ == '__main__':
    base_dir = "/home/t9s9/PycharmProjects/BeeMeter/data/training/"
    dataset = Dataset.from_sqlite(images_dirs=base_dir + "base_training_img", path=base_dir + "base_labels.db",
                                  verbose=False) + \
              Dataset.from_sqlite(images_dirs=base_dir + "gopro_training_img",
                                  path=base_dir + "gopro_labels.db", verbose=False)

    a = Analyser((400, 200), dataset)
    # a.ratio_iou_means(c=3, tries=10, show=True, save=False, verbose=True)
    a.scales()


    def ratios_compare():
        df = pd.read_csv('ratios.csv')
        for i in list(df.groupby(by='dist')):
            plt.plot(i[1]['groups'], i[1]['accuracy'], label=i[0])
        fig = plt.gcf()
        fig.set_size_inches(11, 8)
        plt.legend()
        plt.xlim([1, 8])
        plt.xlabel("Anzahl der Gruppen")
        plt.ylabel("Genauigkeit")
        plt.show()
        fig.savefig('compare_ratio_cluster.png')
