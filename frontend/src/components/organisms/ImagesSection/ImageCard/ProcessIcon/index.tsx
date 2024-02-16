import { useModal } from "../../../../../hooks/contexts/Modal/useModal";
import { components } from "../../../../../services/backend/endpoints";
import { ConfigurationsForm } from "../../../ConfigurationsForm";
import { Icon } from "./Icon";

interface Props {
  image: components["schemas"]["ImageODto"];
  latestProcess: components["schemas"]["ProcessODto"] | undefined;
}

export function ProcessIcon({ image, latestProcess }: Props) {
  const { openModal, closeModal } = useModal();

  if (!latestProcess)
    return (
      <Icon
        type="run"
        onClick={() =>
          openModal(
            <ConfigurationsForm image={image} onSuccessSubmit={closeModal} />
          )
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
            onClick={() => console.log("Implement stop on backend.")}
          />
        );
      case "failed":
        return (
          <Icon
            type="error"
            latestProcess={latestProcess}
            onClick={() => console.log("Implement modal error on frontend.")}
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
