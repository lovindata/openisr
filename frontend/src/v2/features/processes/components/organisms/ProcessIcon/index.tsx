import { ProcessForm } from "@/v2/features/processes/components/organisms/ProcessForm";
import { Icon } from "@/v2/features/processes/components/organisms/ProcessIcon/Icon";
import { useModal } from "@/v2/features/shared/hooks/contexts/Modal/useModal";
import { useBackend } from "@/v2/services/backend";
import { components } from "@/v2/services/backend/endpoints";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  card: components["schemas"]["CardMod"];
}

export function ProcessIcon({ card }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: stopProcess } = useMutation({
    mutationFn: () =>
      backend
        .delete(`/command/v1/images/${card.image_id}/process/stop`)
        .then(() => {}),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["/query/v1/app/cards"] }),
  });

  const { openModal, closeModal } = useModal();

  switch (card.status.type) {
    case "Runnable":
      return (
        <Icon
          type="run"
          onClick={() =>
            openModal(<ProcessForm card={card} onSuccessSubmit={closeModal} />)
          }
        />
      );
    case "Stoppable":
      return (
        <Icon
          type="stop"
          duration={card.status.duration}
          onClick={() => stopProcess()}
        />
      );
    case "Errored":
      return (
        <Icon
          type="error"
          duration={card.status.duration}
          onClick={() =>
            openModal(<ProcessForm card={card} onSuccessSubmit={closeModal} />)
          }
        />
      );
    case "Downloadable":
      return (
        <a href={card.status.image_src} download>
          <Icon type="download" />
        </a>
      );
  }
}
