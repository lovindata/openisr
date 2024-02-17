import { useModal } from "../../../../../hooks/contexts/Modal/useModal";
import { useBackend } from "../../../../../services/backend";
import { components, paths } from "../../../../../services/backend/endpoints";
import { ProcessForm } from "../../../ProcessForm";
import { Icon } from "./Icon";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  image: components["schemas"]["ImageODto"];
  latestProcess: components["schemas"]["ProcessODto"] | undefined;
}

export function ProcessIcon({ image, latestProcess }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: stopProcess } = useMutation({
    mutationFn: () =>
      backend.delete<
        paths["/images/{id}/process"]["delete"]["responses"]["200"]["content"]["application/json"]
      >(`/images/${image.id}/process`),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [`/images/${image.id}/process`],
      });
    },
  });

  const { openModal, closeModal } = useModal();

  if (!latestProcess)
    return (
      <Icon
        type="run"
        onClick={() =>
          openModal(<ProcessForm image={image} onSuccessSubmit={closeModal} />)
        }
      />
    );
  else {
    switch (latestProcess.status.ended?.kind) {
      case undefined:
        return (
          <Icon
            type="stop"
            latestProcess={latestProcess}
            onClick={() => stopProcess()}
          />
        );
      case "failed":
        return (
          <Icon
            type="error"
            latestProcess={latestProcess}
            onClick={() =>
              openModal(
                <ProcessForm
                  image={image}
                  latestProcess={latestProcess}
                  onSuccessSubmit={closeModal}
                />
              )
            }
          />
        );
      case "successful":
        return (
          <a href={image.src.download} download>
            <Icon type="download" latestProcess={latestProcess} />
          </a>
        );
    }
  }
}
