import { useModal } from "../../../hooks/contexts/Modal/useModal";
import { useBackend } from "../../../services/backend";
import { paths } from "../../../services/backend/endpoints";
import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";
import { ConfigurationsForm } from "../ConfigurationsForm";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

interface Props {
  id: number;
  src: string;
  name: string;
  extension: "JPEG" | "PNG" | "WEBP";
  source: { width: number; height: number };
}

export function ImageCard({ id, src, name, extension, source }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: deleteImage } = useMutation({
    mutationFn: () =>
      backend
        .delete<
          paths["/images/{id}"]["delete"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${id}`)
        .then((_) => _.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/images"] }),
  });
  // const { data: latestProcess } = useQuery({
  //   queryKey: [`/images/${id}/process`],
  //   queryFn: () =>
  //     backend
  //       .get<
  //         paths["/images/{id}/process"]["get"]["responses"]["200"]["content"]["application/json"]
  //       >(`/images/${id}/process`)
  //       .then((_) => _.data),
  //   refetchInterval: 5000,
  // });
  const { openModal, closeModal } = useModal();

  return (
    <BorderBox className="flex h-20 w-72 items-center justify-between p-3 text-xs">
      <div className="flex space-x-3">
        <img src={src} alt={name} className="h-14 w-14 rounded-lg" />
        <div className="flex w-32 flex-col justify-between">
          <label className="truncate">{name}</label>
          <span>
            Source: {source.width}x{source.height}px
          </span>
          <span className="font-bold">Target: -</span>
        </div>
      </div>
      <div className="flex items-center space-x-3">
        <SvgIcon
          type="run"
          className="h-6 w-6 cursor-pointer"
          onClick={() =>
            openModal(
              <ConfigurationsForm
                image_id={id}
                initialSource={source}
                initialExtension={extension}
              />
            )
          }
        />
        <SvgIcon
          type="delete"
          className="h-6 w-6 cursor-pointer"
          onClick={() => deleteImage()}
        />
      </div>
    </BorderBox>
  );
}
